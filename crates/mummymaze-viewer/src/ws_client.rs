//! WebSocket client for communicating with the Python model server.
//!
//! Connects to `ws://host:port` and provides a synchronous API that
//! matches the viewer's egui polling model. A single IO thread owns
//! the socket and handles both reads and writes (via a channel).

use mummymaze::event_types::{self, RawTrainingEvent, TrainingEvent};
use mummymaze::game::State;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tungstenite::protocol::Message;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::WebSocket;

type WsStream = WebSocket<MaybeTlsStream<std::net::TcpStream>>;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Server event received from the WebSocket.
#[derive(Debug)]
pub enum ServerEvent {
    Training(TrainingEvent),
    Adversarial(AdversarialEvent),
    Error(String),
}

/// Adversarial loop progress event.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum AdversarialEvent {
    RoundStart {
        round: u32,
        n_rounds: u32,
    },
    GaGeneration {
        round: u32,
        grid_size: u32,
        generation: u32,
        best_fitness: f64,
        avg_fitness: f64,
        solvable_rate: f64,
        pop_size: u32,
    },
    ArchiveUpdate {
        round: u32,
        grid_size: u32,
        n_levels: u32,
        occupancy: u32,
        total_cells: u32,
        time: f64,
    },
    RoundEnd {
        round: u32,
        time: f64,
        ga_levels: u32,
    },
    Done {
        total_ga_levels: u32,
    },
    #[serde(skip)]
    Training(TrainingEvent),
}

/// Result from an evaluate request: per-state action probabilities.
pub struct EvalResult {
    pub probs_by_state: HashMap<State, [f32; 5]>,
}

// ---------------------------------------------------------------------------
// Raw JSON deserialization types
// ---------------------------------------------------------------------------

/// Top-level WebSocket message from server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum RawServerMessage {
    EvaluateResult {
        #[allow(dead_code)]
        request_id: Option<String>,
        states: Vec<RawStateProbs>,
    },
    TrainingEvent {
        event: serde_json::Value,
    },
    AdversarialEvent {
        event: serde_json::Value,
    },
    Error {
        #[allow(dead_code)]
        request_id: Option<String>,
        message: String,
    },
    ReloadResult {
        #[allow(dead_code)]
        request_id: Option<String>,
        #[allow(dead_code)]
        status: String,
    },
    CheckpointsList {
        #[allow(dead_code)]
        request_id: Option<String>,
        checkpoints: Vec<String>,
    },
}

#[derive(Debug, Deserialize)]
struct RawStateProbs {
    state: Vec<i32>,
    probs: Vec<f32>,
}

// ---------------------------------------------------------------------------
// WsClient
// ---------------------------------------------------------------------------

/// WebSocket client for the Python model server.
pub struct WsClient {
    /// Channel to send outgoing messages to the IO thread.
    write_tx: Sender<String>,
    /// Register a oneshot for the next evaluate result.
    eval_register_tx: Mutex<Sender<Sender<Result<EvalResult, String>>>>,
    /// Register a oneshot for the next list_checkpoints result.
    checkpoints_register_tx: Mutex<Sender<Sender<Result<Vec<String>, String>>>>,
    /// Streaming events channel.
    event_rx: Mutex<Receiver<ServerEvent>>,
    /// Connection status.
    connected: Arc<AtomicBool>,
}

impl WsClient {
    /// Connect to the WebSocket server.
    pub fn connect(url: &str) -> Result<Self, String> {
        let (ws, _response) =
            tungstenite::connect(url).map_err(|e| format!("connect failed: {e}"))?;

        // Channels
        let (write_tx, write_rx) = mpsc::channel::<String>();
        let (event_tx, event_rx) = mpsc::channel::<ServerEvent>();
        let (eval_register_tx, eval_register_rx) =
            mpsc::channel::<Sender<Result<EvalResult, String>>>();
        let (ckpt_register_tx, ckpt_register_rx) =
            mpsc::channel::<Sender<Result<Vec<String>, String>>>();

        let connected = Arc::new(AtomicBool::new(true));
        let connected_clone = connected.clone();

        // Spawn IO thread — owns the socket, handles reads and writes
        std::thread::Builder::new()
            .name("ws-io".into())
            .spawn(move || {
                io_loop(ws, write_rx, event_tx, eval_register_rx, ckpt_register_rx);
                connected_clone.store(false, Ordering::Relaxed);
            })
            .map_err(|e| format!("spawn io thread: {e}"))?;

        Ok(Self {
            write_tx,
            eval_register_tx: Mutex::new(eval_register_tx),
            checkpoints_register_tx: Mutex::new(ckpt_register_tx),
            event_rx: Mutex::new(event_rx),
            connected,
        })
    }

    /// Send a raw JSON message.
    fn send(&self, msg: &serde_json::Value) -> Result<(), String> {
        let text = serde_json::to_string(msg).map_err(|e| e.to_string())?;
        self.write_tx
            .send(text)
            .map_err(|_| "io thread gone".to_string())
    }

    /// Evaluate a level by key, blocking until result arrives.
    pub fn evaluate(&self, level_key: &str) -> Result<EvalResult, String> {
        let (tx, rx) = mpsc::channel();
        self.eval_register_tx
            .lock()
            .unwrap()
            .send(tx)
            .map_err(|_| "io thread gone".to_string())?;

        self.send(&serde_json::json!({
            "type": "evaluate",
            "level_key": level_key
        }))?;

        rx.recv().map_err(|_| "io thread gone".to_string())?
    }

    /// Start training with the given config.
    pub fn send_train(&self, config: &serde_json::Value) -> Result<(), String> {
        self.send(&serde_json::json!({
            "type": "train",
            "config": config
        }))
    }

    /// Stop the current training run.
    pub fn send_stop_train(&self) -> Result<(), String> {
        self.send(&serde_json::json!({"type": "stop_train"}))
    }

    /// Start adversarial training loop.
    pub fn send_adversarial(&self, config: &serde_json::Value) -> Result<(), String> {
        self.send(&serde_json::json!({
            "type": "adversarial",
            "config": config
        }))
    }

    /// Stop the adversarial training loop.
    pub fn send_stop_adversarial(&self) -> Result<(), String> {
        self.send(&serde_json::json!({"type": "stop_adversarial"}))
    }

    /// Reload model weights from a checkpoint.
    pub fn send_reload_checkpoint(&self, path: &str) -> Result<(), String> {
        self.send(&serde_json::json!({
            "type": "reload_checkpoint",
            "path": path
        }))
    }

    /// List available checkpoints, blocking until result arrives.
    pub fn list_checkpoints(&self) -> Result<Vec<String>, String> {
        let (tx, rx) = mpsc::channel();
        self.checkpoints_register_tx
            .lock()
            .unwrap()
            .send(tx)
            .map_err(|_| "io thread gone".to_string())?;

        self.send(&serde_json::json!({"type": "list_checkpoints"}))?;

        rx.recv().map_err(|_| "io thread gone".to_string())?
    }

    /// Send shutdown request.
    pub fn send_shutdown(&self) -> Result<(), String> {
        self.send(&serde_json::json!({"type": "shutdown"}))
    }

    /// Non-blocking drain of pending server events.
    pub fn poll_events(&self) -> Vec<ServerEvent> {
        let rx = self.event_rx.lock().unwrap();
        let mut events = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(ev) => events.push(ev),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        events
    }

    /// Check if the connection is alive.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// IO thread — owns the socket, handles reads + writes
// ---------------------------------------------------------------------------

fn io_loop(
    mut ws: WsStream,
    write_rx: Receiver<String>,
    event_tx: Sender<ServerEvent>,
    eval_register_rx: Receiver<Sender<Result<EvalResult, String>>>,
    ckpt_register_rx: Receiver<Sender<Result<Vec<String>, String>>>,
) {
    // Set the underlying TCP stream to non-blocking so we can poll for
    // both incoming messages and outgoing write requests.
    let tcp = match ws.get_ref() {
        MaybeTlsStream::Plain(s) => s,
        _ => {
            eprintln!("ws_client: TLS streams not supported");
            return;
        }
    };
    if let Err(e) = tcp.set_nonblocking(true) {
        eprintln!("ws_client: set_nonblocking: {e}");
        return;
    }

    loop {
        // 1. Flush any pending writes
        loop {
            match write_rx.try_recv() {
                Ok(text) => match ws.send(Message::Text(text.into())) {
                    Ok(()) => {}
                    Err(tungstenite::Error::Io(ref e))
                        if e.kind() == std::io::ErrorKind::WouldBlock =>
                    {
                        // Send buffer full — flush will retry on next iteration
                        break;
                    }
                    Err(e) => {
                        eprintln!("ws_client: send error: {e}");
                        return;
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        // 2. Try to read a message (non-blocking)
        match ws.read() {
            Ok(Message::Text(t)) => {
                dispatch_message(
                    &t,
                    &event_tx,
                    &eval_register_rx,
                    &ckpt_register_rx,
                );
            }
            Ok(Message::Close(_)) => {
                eprintln!("ws_client: server closed connection");
                return;
            }
            Ok(Message::Binary(_)) => {
                eprintln!("ws_client: unexpected binary message");
            }
            Ok(_) => {} // Ping/Pong/Frame
            Err(tungstenite::Error::Io(ref e))
                if e.kind() == std::io::ErrorKind::WouldBlock =>
            {
                // No data available — flush pending writes and sleep briefly
                let _ = ws.flush();
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed) => {
                eprintln!("ws_client: connection closed");
                return;
            }
            Err(e) => {
                eprintln!("ws_client: read error: {e}");
                return;
            }
        }
    }
}

fn dispatch_message(
    text: &str,
    event_tx: &Sender<ServerEvent>,
    eval_register_rx: &Receiver<Sender<Result<EvalResult, String>>>,
    ckpt_register_rx: &Receiver<Sender<Result<Vec<String>, String>>>,
) {
    let raw: RawServerMessage = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ws_client: parse error: {e}: {text}");
            return;
        }
    };

    match raw {
        RawServerMessage::EvaluateResult { states, .. } => {
            let result = parse_eval_result(states);
            if let Ok(tx) = eval_register_rx.try_recv() {
                let _ = tx.send(Ok(result));
            }
        }

        RawServerMessage::TrainingEvent { event } => {
            match serde_json::from_value::<RawTrainingEvent>(event) {
                Ok(raw_ev) => {
                    let ev = event_types::raw_to_training_event(raw_ev);
                    let _ = event_tx.send(ServerEvent::Training(ev));
                }
                Err(e) => {
                    eprintln!("ws_client: bad training event: {e}");
                }
            }
        }

        RawServerMessage::AdversarialEvent { event } => {
            // Try as adversarial event first (consumes value), fall back to training
            match serde_json::from_value::<AdversarialEvent>(event.clone()) {
                Ok(adv) => {
                    let _ = event_tx.send(ServerEvent::Adversarial(adv));
                }
                Err(_) => {
                    if let Ok(raw_train) = serde_json::from_value::<RawTrainingEvent>(event) {
                        let ev = event_types::raw_to_training_event(raw_train);
                        let _ =
                            event_tx.send(ServerEvent::Adversarial(AdversarialEvent::Training(ev)));
                    }
                }
            }
        }

        RawServerMessage::Error { message, .. } => {
            if let Ok(tx) = eval_register_rx.try_recv() {
                let _ = tx.send(Err(message));
            } else if let Ok(tx) = ckpt_register_rx.try_recv() {
                let _ = tx.send(Err(message));
            } else {
                let _ = event_tx.send(ServerEvent::Error(message));
            }
        }

        RawServerMessage::ReloadResult { .. } => {}

        RawServerMessage::CheckpointsList { checkpoints, .. } => {
            if let Ok(tx) = ckpt_register_rx.try_recv() {
                let _ = tx.send(Ok(checkpoints));
            }
        }
    }
}

fn parse_eval_result(states: Vec<RawStateProbs>) -> EvalResult {
    let mut probs_by_state = HashMap::with_capacity(states.len());
    for sp in states {
        if sp.state.len() == 12 && sp.probs.len() == 5 {
            let mut arr = [0i32; 12];
            arr.copy_from_slice(&sp.state);
            let state = State::from_i32_array(&arr);
            let mut probs = [0.0f32; 5];
            probs.copy_from_slice(&sp.probs);
            probs_by_state.insert(state, probs);
        }
    }
    EvalResult { probs_by_state }
}

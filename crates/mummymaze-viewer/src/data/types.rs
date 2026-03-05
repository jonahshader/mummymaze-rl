use mummymaze::batch::LevelAnalysis;
use mummymaze::parse::Level;

pub struct LevelRow {
    pub file_stem: String,
    pub sublevel: usize,
    pub level: Level,
    pub bfs_moves: Option<u32>,
    pub analysis: Option<LevelAnalysis>,
    pub is_duplicate: bool,
    /// Pre-computed lowercase "{file_stem} {sublevel}" for filter matching.
    pub search_text: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortColumn {
    File,
    Sub,
    Grid,
    Bfs,
    States,
    WinProb,
    DeadEnd,
    Safety,
    AgentAcc,
    AgentLoss,
}

impl SortColumn {
    pub(super) fn is_tier2(self) -> bool {
        matches!(
            self,
            SortColumn::States
                | SortColumn::WinProb
                | SortColumn::DeadEnd
                | SortColumn::Safety
                | SortColumn::AgentAcc
                | SortColumn::AgentLoss
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortDir {
    Asc,
    Desc,
}

impl SortDir {
    pub fn toggle(&mut self) {
        *self = match self {
            SortDir::Asc => SortDir::Desc,
            SortDir::Desc => SortDir::Asc,
        };
    }
}

pub struct FilterState {
    pub text: String,
    pub grid_size: Option<i32>,
    pub solvable_only: bool,
    pub show_duplicates: bool,
}

impl Default for FilterState {
    fn default() -> Self {
        FilterState {
            text: String::new(),
            grid_size: None,
            solvable_only: true,
            show_duplicates: false,
        }
    }
}

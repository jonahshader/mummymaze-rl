//! MAP-Elites archive for adversarial level generation.
//!
//! 2D grid indexed by `(bfs_moves, n_states)` bins. Each cell stores the
//! `Individual` closest to `target_log_policy_wp`.

use super::Individual;
use crate::parse::Level;

/// Lightweight snapshot of a single archive cell for visualization.
#[derive(Debug, Clone)]
pub struct ArchiveCell {
    pub log_policy_wp: f64,
    pub bfs_moves: u32,
    pub n_states: usize,
}

/// Snapshot of the full archive grid for the viewer heatmap.
#[derive(Debug, Clone)]
pub struct ArchiveSnapshot {
    pub bfs_bins: usize,
    pub states_bins: usize,
    pub bfs_range: (usize, usize),
    pub states_range: (usize, usize),
    pub target_log_wp: f64,
    /// Flat row-major [bfs_bin][states_bin] grid.
    pub cells: Vec<Option<ArchiveCell>>,
}

/// MAP-Elites archive: 2D grid of (bfs_moves, n_states) bins.
pub struct MapElitesArchive {
    bins: Vec<Vec<Option<Individual>>>,
    bfs_range: (usize, usize),
    states_range: (usize, usize),
    bfs_bins: usize,
    states_bins: usize,
    target_log_wp: f64,
}

impl MapElitesArchive {
    /// Create a new empty archive.
    pub fn new(
        bfs_range: (usize, usize),
        states_range: (usize, usize),
        bfs_bins: usize,
        states_bins: usize,
        target_log_wp: f64,
    ) -> Self {
        let bins = vec![vec![None; states_bins]; bfs_bins];
        MapElitesArchive {
            bins,
            bfs_range,
            states_range,
            bfs_bins,
            states_bins,
            target_log_wp,
        }
    }

    /// Map a raw value to a bin index, clamping to range.
    fn bin_idx(value: usize, range: (usize, usize), n_bins: usize) -> usize {
        if value <= range.0 {
            return 0;
        }
        if value >= range.1 {
            return n_bins - 1;
        }
        let frac = (value - range.0) as f64 / (range.1 - range.0) as f64;
        let idx = (frac * n_bins as f64) as usize;
        idx.min(n_bins - 1)
    }

    /// Try to insert an individual. Returns true if inserted.
    ///
    /// Insertion rule: insert if cell is empty, or if the new individual's
    /// `log_policy_win_prob` is closer to `target_log_wp` than the existing one.
    pub fn try_insert(&mut self, ind: &Individual) -> bool {
        let bi = Self::bin_idx(ind.bfs_moves as usize, self.bfs_range, self.bfs_bins);
        let si = Self::bin_idx(ind.n_states, self.states_range, self.states_bins);

        let new_dist = (ind.log_policy_win_prob - self.target_log_wp).abs();

        let cell = &mut self.bins[bi][si];
        match cell {
            None => {
                *cell = Some(ind.clone());
                true
            }
            Some(existing) => {
                let old_dist = (existing.log_policy_win_prob - self.target_log_wp).abs();
                if new_dist < old_dist {
                    *cell = Some(ind.clone());
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Number of occupied cells and total cells.
    pub fn occupancy(&self) -> (usize, usize) {
        let total = self.bfs_bins * self.states_bins;
        let occupied = self
            .bins
            .iter()
            .flat_map(|row| row.iter())
            .filter(|c| c.is_some())
            .count();
        (occupied, total)
    }

    /// Collect all levels from occupied cells.
    pub fn levels(&self) -> Vec<&Level> {
        self.bins
            .iter()
            .flat_map(|row| row.iter())
            .filter_map(|c| c.as_ref())
            .map(|ind| &ind.level)
            .collect()
    }

    /// Create a lightweight snapshot for visualization.
    pub fn snapshot(&self) -> ArchiveSnapshot {
        let mut cells = Vec::with_capacity(self.bfs_bins * self.states_bins);
        for row in &self.bins {
            for cell in row {
                cells.push(cell.as_ref().map(|ind| ArchiveCell {
                    log_policy_wp: ind.log_policy_win_prob,
                    bfs_moves: ind.bfs_moves,
                    n_states: ind.n_states,
                }));
            }
        }
        ArchiveSnapshot {
            bfs_bins: self.bfs_bins,
            states_bins: self.states_bins,
            bfs_range: self.bfs_range,
            states_range: self.states_range,
            target_log_wp: self.target_log_wp,
            cells,
        }
    }

    /// Collect all individuals from occupied cells.
    pub fn all_individuals(&self) -> Vec<&Individual> {
        self.bins
            .iter()
            .flat_map(|row| row.iter())
            .filter_map(|c| c.as_ref())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::Level;

    fn dummy_level() -> Level {
        Level::from_edges(
            6,
            false,
            &vec![false; 7 * 6],
            &vec![false; 6 * 7],
            "N",
            0,
            (2, 2),
            (4, 4),
            None,
            None,
            &[],
            None,
            None,
        )
    }

    fn make_ind(bfs_moves: u32, n_states: usize, log_policy_wp: f64) -> Individual {
        Individual {
            level: dummy_level(),
            bfs_moves,
            n_states,
            win_prob: 0.5,
            log_policy_win_prob: log_policy_wp,
            fitness: 1.0,
        }
    }

    #[test]
    fn test_insert_empty_cell() {
        let mut archive = MapElitesArchive::new((1, 50), (1, 500), 10, 10, -1.0);
        let ind = make_ind(10, 50, -1.5);
        assert!(archive.try_insert(&ind));
        assert_eq!(archive.occupancy().0, 1);
    }

    #[test]
    fn test_insert_replaces_worse() {
        let mut archive = MapElitesArchive::new((1, 50), (1, 500), 10, 10, -1.0);
        let ind1 = make_ind(10, 50, -2.0); // dist=1.0
        let ind2 = make_ind(10, 50, -1.2); // dist=0.2 (closer to target)
        assert!(archive.try_insert(&ind1));
        assert!(archive.try_insert(&ind2));
        assert_eq!(archive.occupancy().0, 1);
        // The closer one should be stored
        let stored = archive.all_individuals();
        assert!((stored[0].log_policy_win_prob - (-1.2)).abs() < 1e-10);
    }

    #[test]
    fn test_insert_keeps_better() {
        let mut archive = MapElitesArchive::new((1, 50), (1, 500), 10, 10, -1.0);
        let ind1 = make_ind(10, 50, -1.1); // dist=0.1
        let ind2 = make_ind(10, 50, -3.0); // dist=2.0 (worse)
        assert!(archive.try_insert(&ind1));
        assert!(!archive.try_insert(&ind2));
        let stored = archive.all_individuals();
        assert!((stored[0].log_policy_win_prob - (-1.1)).abs() < 1e-10);
    }

    #[test]
    fn test_levels_returns_all() {
        let mut archive = MapElitesArchive::new((1, 50), (1, 500), 10, 10, -1.0);
        // Different bins
        archive.try_insert(&make_ind(5, 10, -1.0));
        archive.try_insert(&make_ind(25, 250, -1.0));
        archive.try_insert(&make_ind(45, 450, -1.0));
        assert_eq!(archive.levels().len(), 3);
    }
}

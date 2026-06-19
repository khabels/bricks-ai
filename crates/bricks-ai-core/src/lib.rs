use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId {
    pub grid: usize,
    pub page: usize,
    pub case: usize,
}

impl NodeId {
    pub fn new(grid: usize, page: usize, case: usize) -> Self {
        Self { grid, page, case }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightState {
    Living,
    Candidate,
    Engraved,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub id: usize,
    pub weight: f32,
    pub signal: f32,
    pub gradient: f32,
    pub state: WeightState,
    pub engraved_weight: Option<f32>,
    pub confidence: f32,
    pub drift_score: f32,
    pub validation_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPage {
    pub id: usize,
    pub parent_page: Option<usize>,
    pub generation: usize,
    pub cases: Vec<Case>,
    pub active: bool,
    pub candidate: bool,
    pub validation_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid {
    pub id: usize,
    pub pages: Vec<GridPage>,
    pub official_page: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correlation {
    pub id: usize,
    pub from: NodeId,
    pub to: NodeId,
    pub coefficient: f32,
    pub coefficient_gradient: f32,
    pub score: f32,
    pub visits: usize,
    pub active: bool,
    pub last_activity: f32,
    pub engraved: bool,
    pub engraved_coefficient: Option<f32>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferredPath {
    pub id: usize,
    pub links: Vec<usize>,
    pub score: f32,
    pub visits: usize,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Group {
    pub id: usize,
    pub name: String,
    pub grids: Vec<usize>,
    pub children: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalZone {
    pub id: usize,
    pub grid: usize,
    pub source_page: usize,
    pub candidate_page: usize,
    pub case_start: usize,
    pub case_end: usize,
    pub best_loss: f32,
    pub validation_score: f32,
    pub active: bool,
    pub validated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub node: NodeId,
    pub expected: f32,
    pub observed: f32,
    pub error: f32,
    pub critical: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AiProvider {
    Ollama,
    OpenAI,
    Anthropic,
    Gemini,
    Mistral,
    Xai,
    DeepSeek,
    Groq,
    Together,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeacherRole {
    Generator,
    Validator,
    Judge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiProviderConfig {
    pub provider: AiProvider,
    pub enabled: bool,
    pub model: String,
    pub api_key_present: bool,
    pub role: TeacherRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherSettings {
    pub providers: Vec<AiProviderConfig>,
    pub consensus_threshold: f32,
    pub min_teacher_confidence: f32,
    pub require_cross_validation: bool,
}

impl TeacherSettings {
    pub fn default_market() -> Self {
        Self {
            providers: vec![
                AiProviderConfig::new(AiProvider::Ollama, "llama3.2", TeacherRole::Generator),
                AiProviderConfig::new(AiProvider::OpenAI, "openai-default", TeacherRole::Generator),
                AiProviderConfig::new(AiProvider::Anthropic, "anthropic-default", TeacherRole::Validator),
                AiProviderConfig::new(AiProvider::Gemini, "gemini-default", TeacherRole::Validator),
                AiProviderConfig::new(AiProvider::Mistral, "mistral-default", TeacherRole::Judge),
                AiProviderConfig::new(AiProvider::Xai, "xai-default", TeacherRole::Validator),
                AiProviderConfig::new(AiProvider::DeepSeek, "deepseek-default", TeacherRole::Generator),
                AiProviderConfig::new(AiProvider::Groq, "groq-default", TeacherRole::Validator),
                AiProviderConfig::new(AiProvider::Together, "together-default", TeacherRole::Generator),
            ],
            consensus_threshold: 0.80,
            min_teacher_confidence: 0.75,
            require_cross_validation: true,
        }
    }
}

impl AiProviderConfig {
    pub fn new(provider: AiProvider, model: &str, role: TeacherRole) -> Self {
        Self {
            provider,
            enabled: false,
            model: model.to_string(),
            api_key_present: false,
            role,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherDataCandidate {
    pub id: usize,
    pub prompt: String,
    pub expected_answer: String,
    pub source_provider: AiProvider,
    pub teacher_confidence: f32,
    pub validator_score: f32,
    pub accepted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackCoverageMode {
    Manual,
    Broad,
    Universal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackStatus {
    Idle,
    Running,
    Paused,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyLevel {
    Normal,
    HighStakes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeNode {
    pub name: String,
    pub enabled: bool,
    pub safety: SafetyLevel,
    pub children: Vec<ThemeNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPack {
    pub name: String,
    pub enabled: bool,
    pub coverage_mode: PackCoverageMode,
    pub max_depth: usize,
    pub themes: Vec<ThemeNode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackJobKind {
    ExpandTheme,
    GenerateTrainingData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackJob {
    pub id: usize,
    pub kind: PackJobKind,
    pub theme_path: Vec<String>,
    pub depth: usize,
    pub requested_items: usize,
    pub safety: SafetyLevel,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PackTrainingState {
    pub status: PackStatus,
    pub queue: VecDeque<PackJob>,
    pub current_job: Option<PackJob>,
    pub accepted_items: usize,
    pub rejected_items: usize,
    pub trained_items: usize,
    pub max_items_per_theme: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedSubTheme {
    pub name: String,
    pub safety: SafetyLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackTeacherItem {
    pub question: String,
    pub answer: String,
    pub confidence: f32,
    pub tags: Vec<String>,
    pub verification_notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedPackItem {
    pub theme_path: Vec<String>,
    pub question: String,
    pub answer: String,
    pub teacher_confidence: f32,
    pub validator_score: f32,
    pub accepted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PackTickAction {
    Idle,
    Finished,
    NeedSubthemeExpansion { job: PackJob, prompt: String },
    NeedTrainingData { job: PackJob, prompt: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngravedCase {
    pub grid: usize,
    pub page: usize,
    pub case: usize,
    pub weight: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngravedCorrelation {
    pub from: NodeId,
    pub to: NodeId,
    pub coefficient: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelDimensionState {
    pub id: usize,
    pub seed: u64,
    pub races: usize,
    pub wins: usize,
    pub total_loss: f32,
    pub best_loss: f32,
    pub total_score: f32,
    pub cross_validations: usize,
    pub paths_found: usize,
    pub convergence_accepts: usize,
}

impl ParallelDimensionState {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            seed: dimension_seed(id),
            races: 0,
            wins: 0,
            total_loss: 0.0,
            best_loss: f32::MAX,
            total_score: 0.0,
            cross_validations: 0,
            paths_found: 0,
            convergence_accepts: 0,
        }
    }

    pub fn avg_loss(&self) -> f32 {
        if self.races == 0 { 0.0 } else { self.total_loss / self.races as f32 }
    }

    pub fn avg_score(&self) -> f32 {
        if self.races == 0 { 0.0 } else { self.total_score / self.races as f32 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionPathRecord {
    pub dimension_id: usize,
    pub path_key: String,
    pub input: NodeId,
    pub mid_1: NodeId,
    pub mid_2: NodeId,
    pub output: NodeId,
    pub visits: usize,
    pub wins: usize,
    pub loss: f32,
    pub score: f32,
    pub best_loss: f32,
    pub agreement_count: usize,
    pub cross_validated: bool,
    pub convergence_score: f32,
    pub convergence_winner: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionRaceCandidate {
    pub dimension_id: usize,
    pub path_key: String,
    pub input: NodeId,
    pub mid_1: NodeId,
    pub mid_2: NodeId,
    pub output: NodeId,
    pub loss: f32,
    pub score: f32,
    pub agreement_count: usize,
    pub cross_validated: bool,
    pub convergence_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceClusterRecord {
    pub cluster_id: usize,
    pub final_node: NodeId,
    pub representative_case: usize,
    pub member_cases: Vec<usize>,
    pub candidate_count: usize,
    pub vote_weight: f32,
    pub avg_loss: f32,
    pub avg_candidate_score: f32,
    pub convergence_score: f32,
    pub contributing_dimensions: Vec<usize>,
    pub contributing_paths: Vec<String>,
    pub neighbor_merges: usize,
    pub supporting_paths_reinforced: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCubeState {
    pub enabled: bool,
    pub total_candidates_received: usize,
    pub total_clusters_built: usize,
    pub total_winner_votes: usize,
    pub total_neighbor_merges: usize,
    pub total_supporting_paths_reinforced: usize,
    pub winner_history: Vec<ConvergenceClusterRecord>,
    pub last_winner: Option<ConvergenceClusterRecord>,
}

impl Default for ConvergenceCubeState {
    fn default() -> Self {
        Self {
            enabled: true,
            total_candidates_received: 0,
            total_clusters_built: 0,
            total_winner_votes: 0,
            total_neighbor_merges: 0,
            total_supporting_paths_reinforced: 0,
            winner_history: Vec::new(),
            last_winner: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFinalDestructionEvent {
    pub reason: String,
    pub path_key: String,
    pub cases_destroyed: usize,
    pub correlations_destroyed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFinalDestructionState {
    pub enabled: bool,
    pub runs: usize,
    pub candidates_seen: usize,
    pub candidates_forwarded: usize,
    pub candidates_destroyed: usize,
    pub candidates_rescued: usize,
    pub cases_destroyed: usize,
    pub correlations_destroyed: usize,
    pub blocks_destroyed: usize,
    pub events: Vec<PreFinalDestructionEvent>,
}

impl Default for PreFinalDestructionState {
    fn default() -> Self {
        Self {
            enabled: true,
            runs: 0,
            candidates_seen: 0,
            candidates_forwarded: 0,
            candidates_destroyed: 0,
            candidates_rescued: 0,
            cases_destroyed: 0,
            correlations_destroyed: 0,
            blocks_destroyed: 0,
            events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngravedModel {
    pub cases: Vec<EngravedCase>,
    pub correlations: Vec<EngravedCorrelation>,
    #[serde(default)]
    pub dimensions: Vec<ParallelDimensionState>,
    #[serde(default)]
    pub dimension_paths: Vec<DimensionPathRecord>,
    #[serde(default)]
    pub convergence_cube: ConvergenceCubeState,
    #[serde(default)]
    pub pre_final_destruction: PreFinalDestructionState,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RawAITrainer {
    pub grids: Vec<Grid>,
    pub correlations: Vec<Correlation>,
    pub adjacency: HashMap<NodeId, Vec<usize>>,
    pub paths: Vec<PreferredPath>,
    pub groups: Vec<Group>,
    pub critical_zones: Vec<CriticalZone>,
    pub teacher_settings: TeacherSettings,
    pub candidate_dataset: Vec<TeacherDataCandidate>,
    pub training_pack: TrainingPack,
    pub pack_state: PackTrainingState,
    #[serde(default)]
    pub dimensions: Vec<ParallelDimensionState>,
    #[serde(default)]
    pub dimension_paths: Vec<DimensionPathRecord>,
    #[serde(default)]
    pub convergence_cube: ConvergenceCubeState,
    #[serde(default)]
    pub pre_final_destruction: PreFinalDestructionState,
    pub case_learning_rate: f32,
    pub correlation_learning_rate: f32,
    pub affinity_learning_rate: f32,
    pub prune_threshold: f32,
    pub max_coefficient: f32,
}

impl RawAITrainer {
    pub fn new(grid_count: usize, cases_per_grid: usize) -> Self {
        let mut grids = Vec::new();

        for grid_id in 0..grid_count {
            let mut cases = Vec::new();
            for case_id in 0..cases_per_grid {
                cases.push(Case {
                    id: case_id,
                    weight: 1.0,
                    signal: 0.0,
                    gradient: 0.0,
                    state: WeightState::Living,
                    engraved_weight: None,
                    confidence: 0.0,
                    drift_score: 0.0,
                    validation_score: 0.0,
                });
            }

            let page_0 = GridPage {
                id: 0,
                parent_page: None,
                generation: 0,
                cases,
                active: true,
                candidate: false,
                validation_score: 0.0,
            };

            grids.push(Grid {
                id: grid_id,
                pages: vec![page_0],
                official_page: 0,
            });
        }

        Self {
            grids,
            correlations: Vec::new(),
            adjacency: HashMap::new(),
            paths: Vec::new(),
            groups: Vec::new(),
            critical_zones: Vec::new(),
            teacher_settings: TeacherSettings::default_market(),
            candidate_dataset: Vec::new(),
            training_pack: universal_root_pack(),
            pack_state: PackTrainingState {
                status: PackStatus::Idle,
                queue: VecDeque::new(),
                current_job: None,
                accepted_items: 0,
                rejected_items: 0,
                trained_items: 0,
                max_items_per_theme: 10,
            },
            dimensions: Vec::new(),
            dimension_paths: Vec::new(),
            convergence_cube: ConvergenceCubeState::default(),
            pre_final_destruction: PreFinalDestructionState::default(),
            case_learning_rate: 0.01,
            correlation_learning_rate: 0.01,
            affinity_learning_rate: 0.001,
            prune_threshold: 0.0001,
            max_coefficient: 3.0,
        }
    }

    pub fn official_node(&self, grid: usize, case: usize) -> NodeId {
        NodeId::new(grid, self.grids[grid].official_page, case)
    }

    pub fn add_group(&mut self, name: &str, grids: Vec<usize>, children: Vec<usize>) -> usize {
        let id = self.groups.len();
        self.groups.push(Group { id, name: name.to_string(), grids, children });
        id
    }

    pub fn add_correlation(&mut self, from: NodeId, to: NodeId, coefficient: f32) -> usize {
        self.assert_valid_node(from);
        self.assert_valid_node(to);
        let id = self.correlations.len();
        self.correlations.push(Correlation {
            id,
            from,
            to,
            coefficient,
            coefficient_gradient: 0.0,
            score: 0.0,
            visits: 0,
            active: true,
            last_activity: 0.0,
            engraved: false,
            engraved_coefficient: None,
            confidence: 0.0,
        });
        self.adjacency.entry(from).or_default().push(id);
        id
    }

    pub fn train_step(&mut self, inputs: &[(NodeId, f32)], targets: &[(NodeId, f32)], forward_passes: usize) -> f32 {
        self.reset_temporary_state();
        self.inject_inputs(inputs);
        let used_links = self.forward(forward_passes);
        let loss = self.compute_loss_and_seed_error(targets);
        self.backward_error();
        self.apply_gradients();
        let reward = 1.0 / (1.0 + loss);
        self.reinforce_used_links(&used_links, reward);
        self.increase_confidence(&used_links, reward);
        self.score_preferred_paths(&used_links, reward);
        loss
    }

    pub fn predict(&mut self, inputs: &[(NodeId, f32)], outputs: &[NodeId], forward_passes: usize) -> Vec<f32> {
        self.reset_temporary_state();
        self.inject_inputs(inputs);
        self.forward(forward_passes);
        outputs.iter().map(|node| self.signal(*node)).collect()
    }

    pub fn forward(&mut self, passes: usize) -> HashSet<usize> {
        let mut used_links = HashSet::new();
        for _ in 0..passes {
            for link_id in 0..self.correlations.len() {
                if !self.correlations[link_id].active {
                    continue;
                }
                let from = self.correlations[link_id].from;
                let to = self.correlations[link_id].to;
                let coefficient = self.correlations[link_id].coefficient;
                let influence = self.signal(from) * self.weight(from) * coefficient;
                if influence.abs() > 1e-9 {
                    self.add_signal(to, influence);
                    self.correlations[link_id].last_activity = influence.abs();
                    used_links.insert(link_id);
                }
            }
        }
        used_links
    }

    pub fn compute_loss_and_seed_error(&mut self, targets: &[(NodeId, f32)]) -> f32 {
        let mut loss = 0.0;
        for (node, expected) in targets {
            let observed = self.signal(*node);
            let error = expected - observed;
            loss += 0.5 * error * error;
            self.add_case_gradient(*node, error);
        }
        loss
    }

    pub fn backward_error(&mut self) {
        for link_id in (0..self.correlations.len()).rev() {
            if !self.correlations[link_id].active {
                continue;
            }
            let from = self.correlations[link_id].from;
            let to = self.correlations[link_id].to;
            let coefficient = self.correlations[link_id].coefficient;
            let target_error = self.case_gradient(to);
            let source_signal = self.signal(from);
            let source_weight = self.weight(from);
            let coefficient_gradient = target_error * source_signal * source_weight;
            let source_gradient = target_error * coefficient;
            self.correlations[link_id].coefficient_gradient += coefficient_gradient;
            self.add_case_gradient(from, source_gradient);
        }
    }

    pub fn can_update_case(&self, node: NodeId) -> bool {
        let page = &self.grids[node.grid].pages[node.page];
        if page.candidate {
            return self.critical_zones.iter().any(|zone| {
                zone.active
                    && node.grid == zone.grid
                    && node.page == zone.candidate_page
                    && node.case >= zone.case_start
                    && node.case <= zone.case_end
            });
        }
        let case = &page.cases[node.case];
        matches!(case.state, WeightState::Living | WeightState::Candidate)
    }

    pub fn apply_gradients(&mut self) {
        for grid_id in 0..self.grids.len() {
            for page_id in 0..self.grids[grid_id].pages.len() {
                for case_id in 0..self.grids[grid_id].pages[page_id].cases.len() {
                    let node = NodeId::new(grid_id, page_id, case_id);
                    if !self.can_update_case(node) {
                        if let Some(saved) = self.grids[grid_id].pages[page_id].cases[case_id].engraved_weight {
                            self.grids[grid_id].pages[page_id].cases[case_id].weight = saved;
                        }
                        continue;
                    }
                    let case = &mut self.grids[grid_id].pages[page_id].cases[case_id];
                    case.weight += self.case_learning_rate * case.gradient;
                    case.weight = case.weight.clamp(-10.0, 10.0);
                }
            }
        }

        for correlation in &mut self.correlations {
            if correlation.engraved {
                if let Some(saved) = correlation.engraved_coefficient {
                    correlation.coefficient = saved;
                }
                continue;
            }
            correlation.coefficient += self.correlation_learning_rate * correlation.coefficient_gradient;
            correlation.coefficient = correlation.coefficient.clamp(-self.max_coefficient, self.max_coefficient);
        }
    }

    pub fn reinforce_used_links(&mut self, used_links: &HashSet<usize>, reward: f32) {
        for link_id in used_links {
            let link = &mut self.correlations[*link_id];
            if link.engraved {
                continue;
            }
            link.visits += 1;
            link.score = 0.95 * link.score + 0.05 * reward;
            let direction = if link.coefficient >= 0.0 { 1.0 } else { -1.0 };
            link.coefficient += self.affinity_learning_rate * reward * link.last_activity * direction;
            link.coefficient = link.coefficient.clamp(-self.max_coefficient, self.max_coefficient);
        }
    }

    pub fn increase_confidence(&mut self, used_links: &HashSet<usize>, reward: f32) {
        for link_id in used_links {
            let from = self.correlations[*link_id].from;
            let to = self.correlations[*link_id].to;
            self.add_confidence(from, reward);
            self.add_confidence(to, reward);
        }
    }

    pub fn reinforce_node_confidence(&mut self, node: NodeId, reward: f32, repeats: usize) {
        self.assert_valid_node(node);
        for _ in 0..repeats.max(1) {
            self.add_confidence(node, reward.clamp(0.0, 1.0));
        }
    }

    pub fn reinforce_path_correlations(
        &mut self,
        nodes: &[NodeId],
        reward: f32,
        repeats: usize,
        score_boost: f32,
        coefficient_boost: f32,
        survival_min_visits: usize,
        survival_bonus: f32,
    ) -> usize {
        if nodes.len() < 2 {
            return 0;
        }

        let reward = reward.clamp(0.0, 1.0);
        let repeats = repeats.clamp(1, 64);
        let score_step = (0.05 * score_boost.max(0.0)).clamp(0.01, 0.35);
        let confidence_step = (0.035 * score_boost.max(0.0)).clamp(0.005, 0.25);
        let coefficient_boost = coefficient_boost.max(0.0);
        let survival_min_visits = survival_min_visits.max(1);
        let survival_bonus = survival_bonus.clamp(0.0, 0.10);
        let mut touched = 0usize;

        for pair in nodes.windows(2) {
            let Some(link_id) = self.active_correlation_between(pair[0], pair[1]) else {
                continue;
            };
            touched += 1;

            for _ in 0..repeats {
                let link = &mut self.correlations[link_id];
                if !link.active {
                    break;
                }

                link.visits += 1;
                link.score = (link.score + score_step * (reward - link.score)).clamp(0.0, 1.0);
                link.confidence = (link.confidence + confidence_step * (reward - link.confidence)).clamp(0.0, 1.0);

                if link.visits >= survival_min_visits {
                    link.score = (link.score + survival_bonus * reward).clamp(0.0, 1.0);
                    link.confidence = (link.confidence + 0.5 * survival_bonus * reward).clamp(0.0, 1.0);
                }

                if !link.engraved {
                    let direction = if link.coefficient >= 0.0 { 1.0 } else { -1.0 };
                    let activity = link.last_activity.max(0.25);
                    link.coefficient += self.affinity_learning_rate * coefficient_boost * reward * activity * direction;
                    link.coefficient = link.coefficient.clamp(-self.max_coefficient, self.max_coefficient);
                }
            }
        }

        touched
    }

    fn active_correlation_between(&self, from: NodeId, to: NodeId) -> Option<usize> {
        self.adjacency.get(&from)?.iter().copied().find(|link_id| {
            let link = &self.correlations[*link_id];
            link.active && link.to == to
        })
    }

    pub fn prune_weak_links(&mut self, min_score: f32) {
        for link in &mut self.correlations {
            if link.engraved {
                continue;
            }
            if link.coefficient.abs() < self.prune_threshold && link.score < min_score {
                link.active = false;
            }
        }
        for path in &mut self.paths {
            let contains_dead_link = path.links.iter().any(|id| !self.correlations[*id].active);
            if contains_dead_link || path.score < min_score {
                path.active = false;
            }
        }
    }

    pub fn discover_paths_from(&mut self, start: NodeId, max_depth: usize, beam_width: usize) {
        self.assert_valid_node(start);
        let mut frontier: Vec<Vec<usize>> = self
            .adjacency
            .get(&start)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|link_id| vec![link_id])
            .collect();

        for _depth in 1..=max_depth {
            let mut next_frontier = Vec::new();
            for path in &frontier {
                if path.len() >= 2 {
                    self.save_path_if_new(path.clone());
                }
                let last_link_id = *path.last().unwrap();
                let last_node = self.correlations[last_link_id].to;
                let mut candidates = self.adjacency.get(&last_node).cloned().unwrap_or_default();
                candidates.retain(|link_id| self.correlations[*link_id].active);
                candidates.sort_by(|a, b| self.link_quality(*b).partial_cmp(&self.link_quality(*a)).unwrap_or(std::cmp::Ordering::Equal));
                for next_link_id in candidates.into_iter().take(beam_width) {
                    if self.path_would_cycle(path, next_link_id) {
                        continue;
                    }
                    let mut extended = path.clone();
                    extended.push(next_link_id);
                    next_frontier.push(extended);
                }
            }
            next_frontier.sort_by(|a, b| self.path_quality(b).partial_cmp(&self.path_quality(a)).unwrap_or(std::cmp::Ordering::Equal));
            next_frontier.truncate(beam_width);
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
    }

    pub fn score_preferred_paths(&mut self, used_links: &HashSet<usize>, reward: f32) {
        for path in &mut self.paths {
            if !path.active {
                continue;
            }
            let fully_used = path.links.iter().all(|link_id| used_links.contains(link_id));
            if fully_used {
                path.visits += 1;
                path.score = 0.95 * path.score + 0.05 * reward;
            } else {
                path.score *= 0.999;
            }
        }
    }

    pub fn engrave_validated_weights(&mut self, min_case_confidence: f32, min_link_score: f32) {
        for grid in &mut self.grids {
            let page_id = grid.official_page;
            for case in &mut grid.pages[page_id].cases {
                if case.confidence >= min_case_confidence {
                    case.state = WeightState::Engraved;
                    case.engraved_weight = Some(case.weight);
                }
            }
        }
        for correlation in &mut self.correlations {
            if correlation.score >= min_link_score && correlation.active {
                correlation.engraved = true;
                correlation.engraved_coefficient = Some(correlation.coefficient);
                correlation.confidence = correlation.score;
            }
        }
    }

    pub fn export_engraved_model(&self) -> EngravedModel {
        let mut cases = Vec::new();
        let mut correlations = Vec::new();
        for grid in &self.grids {
            for page in &grid.pages {
                for case in &page.cases {
                    if let Some(weight) = case.engraved_weight {
                        cases.push(EngravedCase { grid: grid.id, page: page.id, case: case.id, weight, confidence: case.confidence });
                    }
                }
            }
        }
        for correlation in &self.correlations {
            if let Some(coefficient) = correlation.engraved_coefficient {
                correlations.push(EngravedCorrelation { from: correlation.from, to: correlation.to, coefficient, score: correlation.score });
            }
        }
        EngravedModel {
            cases,
            correlations,
            dimensions: self.dimensions.clone(),
            dimension_paths: self.dimension_paths.clone(),
            convergence_cube: self.convergence_cube.clone(),
            pre_final_destruction: self.pre_final_destruction.clone(),
        }
    }

    pub fn ensure_parallel_dimensions(&mut self, count: usize) {
        while self.dimensions.len() < count {
            let id = self.dimensions.len();
            self.dimensions.push(ParallelDimensionState::new(id));
        }
        self.convergence_cube.enabled = count > 1;
    }

    pub fn record_dimension_convergence(
        &mut self,
        dimension_count: usize,
        candidates: &[DimensionRaceCandidate],
        winner: ConvergenceClusterRecord,
    ) {
        self.ensure_parallel_dimensions(dimension_count);
        let contributing_paths: HashSet<String> = winner.contributing_paths.iter().cloned().collect();
        let winner_dim = winner.contributing_dimensions.first().copied();

        for candidate in candidates {
            self.ensure_parallel_dimensions(candidate.dimension_id + 1);
            let state = &mut self.dimensions[candidate.dimension_id];
            state.races += 1;
            state.total_loss += candidate.loss;
            state.best_loss = state.best_loss.min(candidate.loss);
            state.total_score += candidate.score;
            if candidate.cross_validated {
                state.cross_validations += 1;
            }
            if contributing_paths.contains(&candidate.path_key) {
                state.convergence_accepts += 1;
            }
            if Some(candidate.dimension_id) == winner_dim {
                state.wins += 1;
            }

            let mut found_existing = false;
            for record in &mut self.dimension_paths {
                if record.path_key == candidate.path_key {
                    record.visits += 1;
                    record.loss = candidate.loss;
                    record.score = candidate.score;
                    record.best_loss = record.best_loss.min(candidate.loss);
                    record.agreement_count = candidate.agreement_count;
                    record.cross_validated |= candidate.cross_validated;
                    record.convergence_score = candidate.convergence_score;
                    if contributing_paths.contains(&candidate.path_key) {
                        record.convergence_winner = true;
                    }
                    if Some(candidate.dimension_id) == winner_dim {
                        record.wins += 1;
                    }
                    found_existing = true;
                    break;
                }
            }
            if !found_existing {
                self.dimension_paths.push(DimensionPathRecord {
                    dimension_id: candidate.dimension_id,
                    path_key: candidate.path_key.clone(),
                    input: candidate.input,
                    mid_1: candidate.mid_1,
                    mid_2: candidate.mid_2,
                    output: candidate.output,
                    visits: 1,
                    wins: if Some(candidate.dimension_id) == winner_dim { 1 } else { 0 },
                    loss: candidate.loss,
                    score: candidate.score,
                    best_loss: candidate.loss,
                    agreement_count: candidate.agreement_count,
                    cross_validated: candidate.cross_validated,
                    convergence_score: candidate.convergence_score,
                    convergence_winner: contributing_paths.contains(&candidate.path_key),
                });
                self.dimensions[candidate.dimension_id].paths_found += 1;
            }
        }

        self.convergence_cube.total_candidates_received += candidates.len();
        self.convergence_cube.total_clusters_built += 1;
        self.convergence_cube.total_winner_votes += winner.candidate_count;
        self.convergence_cube.total_neighbor_merges += winner.neighbor_merges;
        self.convergence_cube.total_supporting_paths_reinforced += winner.supporting_paths_reinforced;
        self.convergence_cube.last_winner = Some(winner.clone());
        self.convergence_cube.winner_history.push(winner);
        if self.convergence_cube.winner_history.len() > 2048 {
            let overflow = self.convergence_cube.winner_history.len() - 2048;
            self.convergence_cube.winner_history.drain(0..overflow);
        }
    }

    pub fn destroy_prefinal_candidate_path(
        &mut self,
        path_key: &str,
        nodes: &[NodeId],
        reason: &str,
        protect_confidence: f32,
    ) -> (usize, usize) {
        let mut cases_destroyed = 0usize;
        let mut correlations_destroyed = 0usize;

        for pair in nodes.windows(2) {
            let from = pair[0];
            let to = pair[1];
            for link in &mut self.correlations {
                if link.from == from && link.to == to && link.active && !link.engraved {
                    link.active = false;
                    link.score = 0.0;
                    link.coefficient = 0.0;
                    link.coefficient_gradient = 0.0;
                    link.confidence = 0.0;
                    correlations_destroyed += 1;
                }
            }
        }

        for node in nodes.iter().skip(1).copied() {
            if let Some(grid) = self.grids.get_mut(node.grid) {
                if let Some(page) = grid.pages.get_mut(node.page) {
                    if let Some(case) = page.cases.get_mut(node.case) {
                        if case.confidence <= protect_confidence && case.state != WeightState::Engraved {
                            case.weight = 1.0;
                            case.signal = 0.0;
                            case.gradient = 0.0;
                            case.state = WeightState::Living;
                            case.engraved_weight = None;
                            case.confidence = 0.0;
                            case.drift_score = 0.0;
                            case.validation_score = 0.0;
                            cases_destroyed += 1;
                        }
                    }
                }
            }
        }

        if correlations_destroyed > 0 {
            for path in &mut self.paths {
                if path.links.iter().any(|id| !self.correlations[*id].active) {
                    path.active = false;
                }
            }
        }

        self.pre_final_destruction.candidates_destroyed += 1;
        self.pre_final_destruction.blocks_destroyed += 1;
        self.pre_final_destruction.cases_destroyed += cases_destroyed;
        self.pre_final_destruction.correlations_destroyed += correlations_destroyed;
        self.pre_final_destruction.events.push(PreFinalDestructionEvent {
            reason: reason.to_string(),
            path_key: path_key.to_string(),
            cases_destroyed,
            correlations_destroyed,
        });
        if self.pre_final_destruction.events.len() > 512 {
            let overflow = self.pre_final_destruction.events.len() - 512;
            self.pre_final_destruction.events.drain(0..overflow);
        }

        (cases_destroyed, correlations_destroyed)
    }

    pub fn save_engraved_model(&self, path: &str) -> io::Result<()> {
        fs::write(path, self.export_engraved_model().to_json())
    }

    pub fn detect_drift(&mut self, targets: &[(NodeId, f32)], drift_threshold: f32) -> Vec<DriftReport> {
        let mut reports = Vec::new();
        for (node, expected) in targets {
            let observed = self.signal(*node);
            let error = expected - observed;
            let critical = error.abs() >= drift_threshold;
            if critical {
                let case = &mut self.grids[node.grid].pages[node.page].cases[node.case];
                case.drift_score = 0.90 * case.drift_score + 0.10 * error.abs();
                case.state = WeightState::Critical;
            }
            reports.push(DriftReport { node: *node, expected: *expected, observed, error, critical });
        }
        reports
    }

    pub fn fork_page_candidate(&mut self, grid_id: usize, source_page_id: usize) -> usize {
        let source_page = self.grids[grid_id].pages[source_page_id].clone();
        let new_page_id = self.grids[grid_id].pages.len();
        let mut new_page = source_page;
        new_page.id = new_page_id;
        new_page.parent_page = Some(source_page_id);
        new_page.generation += 1;
        new_page.active = true;
        new_page.candidate = true;
        new_page.validation_score = 0.0;
        for case in &mut new_page.cases {
            case.state = WeightState::Candidate;
            case.signal = 0.0;
            case.gradient = 0.0;
            case.engraved_weight = None;
            case.confidence *= 0.75;
            case.validation_score = 0.0;
        }
        self.grids[grid_id].pages.push(new_page);
        new_page_id
    }

    pub fn create_critical_zone(&mut self, grid: usize, page: usize, center_case: usize, radius: usize) -> usize {
        let case_count = self.grids[grid].pages[page].cases.len();
        let case_start = center_case.saturating_sub(radius);
        let case_end = (center_case + radius).min(case_count - 1);
        let candidate_page = self.fork_page_candidate(grid, page);
        let zone_id = self.critical_zones.len();
        self.critical_zones.push(CriticalZone {
            id: zone_id,
            grid,
            source_page: page,
            candidate_page,
            case_start,
            case_end,
            best_loss: f32::MAX,
            validation_score: 0.0,
            active: true,
            validated: false,
        });
        zone_id
    }

    pub fn validate_candidate_zone(&mut self, zone_id: usize, old_loss: f32, new_loss: f32, validation_margin: f32) -> bool {
        let zone = &mut self.critical_zones[zone_id];
        if new_loss < zone.best_loss {
            zone.best_loss = new_loss;
        }
        let improvement = old_loss - new_loss;
        if improvement > 0.0 {
            zone.validation_score = 0.90 * zone.validation_score + 0.10 * improvement;
        } else {
            zone.validation_score *= 0.95;
        }
        zone.validation_score >= validation_margin
    }

    pub fn promote_candidate_zone(&mut self, zone_id: usize) {
        let zone = self.critical_zones[zone_id].clone();
        let grid = &mut self.grids[zone.grid];
        grid.official_page = zone.candidate_page;
        let candidate_page = &mut grid.pages[zone.candidate_page];
        candidate_page.candidate = false;
        candidate_page.active = true;
        candidate_page.validation_score = zone.validation_score;
        for case_id in zone.case_start..=zone.case_end {
            let case = &mut candidate_page.cases[case_id];
            case.state = WeightState::Engraved;
            case.engraved_weight = Some(case.weight);
            case.confidence = (case.confidence + zone.validation_score).clamp(0.0, 1.0);
        }
        self.critical_zones[zone_id].active = false;
        self.critical_zones[zone_id].validated = true;
    }

    pub fn retry_zone_with_new_page(&mut self, zone_id: usize) -> usize {
        let old_zone = self.critical_zones[zone_id].clone();
        self.critical_zones[zone_id].active = false;
        let new_candidate_page = self.fork_page_candidate(old_zone.grid, old_zone.source_page);
        let new_zone_id = self.critical_zones.len();
        self.critical_zones.push(CriticalZone {
            id: new_zone_id,
            grid: old_zone.grid,
            source_page: old_zone.source_page,
            candidate_page: new_candidate_page,
            case_start: old_zone.case_start,
            case_end: old_zone.case_end,
            best_loss: f32::MAX,
            validation_score: 0.0,
            active: true,
            validated: false,
        });
        new_zone_id
    }

    pub fn accept_teacher_candidate(&mut self, mut candidate: TeacherDataCandidate) -> bool {
        let accepted = candidate.teacher_confidence >= self.teacher_settings.min_teacher_confidence
            && (!self.teacher_settings.require_cross_validation
                || candidate.validator_score >= self.teacher_settings.consensus_threshold);
        candidate.accepted = accepted;
        self.candidate_dataset.push(candidate);
        accepted
    }

    pub fn train_from_teacher_candidate(&mut self, candidate_id: usize, input_node: NodeId, output_node: NodeId, forward_passes: usize) -> Option<f32> {
        let candidate = self.candidate_dataset.get(candidate_id)?;
        if !candidate.accepted {
            return None;
        }
        let input_value = hash_text_to_signal(&candidate.prompt);
        let expected_value = hash_text_to_signal(&candidate.expected_answer);
        Some(self.train_step(&[(input_node, input_value)], &[(output_node, expected_value)], forward_passes))
    }

    pub fn start_pack_training(&mut self) {
        self.pack_state.queue.clear();
        self.pack_state.status = PackStatus::Running;
        self.pack_state.current_job = None;
        for (id, theme) in self.training_pack.themes.iter().filter(|theme| theme.enabled).enumerate() {
            let kind = if self.training_pack.max_depth == 0 {
                PackJobKind::GenerateTrainingData
            } else {
                PackJobKind::ExpandTheme
            };

            self.pack_state.queue.push_back(PackJob {
                id,
                kind,
                theme_path: vec![theme.name.clone()],
                depth: 0,
                requested_items: self.pack_state.max_items_per_theme,
                safety: theme.safety,
            });
        }
    }

    pub fn pack_training_tick(&mut self) -> PackTickAction {
        if self.pack_state.status != PackStatus::Running {
            return PackTickAction::Idle;
        }
        let Some(job) = self.pack_state.queue.pop_front() else {
            self.pack_state.status = PackStatus::Finished;
            return PackTickAction::Finished;
        };
        self.pack_state.current_job = Some(job.clone());
        match job.kind {
            PackJobKind::ExpandTheme => {
                let max_children = std::env::var("BRICKS_AI_SUBTHEMES_PER_THEME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(8)
                    .clamp(1, 20);
                let prompt = build_subtheme_expansion_prompt(&job.theme_path, max_children);
                PackTickAction::NeedSubthemeExpansion { job, prompt }
            }
            PackJobKind::GenerateTrainingData => {
                let prompt = build_teacher_prompt(&job);
                PackTickAction::NeedTrainingData { job, prompt }
            }
        }
    }

    pub fn accept_generated_subthemes(&mut self, parent_job: &PackJob, subthemes: Vec<GeneratedSubTheme>) {
        // Data-first scheduling: every expanded theme immediately gets a training-data job.
        // Older versions pushed the data job behind all pending expansion jobs. With a deep
        // pack, a short run could spend all requested steps expanding the tree and finish with
        // trained_items=0. Bricks training steps must produce model updates as early as possible.
        self.enqueue_data_generation_front(parent_job);

        if parent_job.depth >= self.training_pack.max_depth {
            return;
        }

        for (next_id, subtheme) in (self.next_pack_job_id()..).zip(subthemes.into_iter().rev()) {
            let mut new_path = parent_job.theme_path.clone();
            new_path.push(subtheme.name);
            self.pack_state.queue.push_back(PackJob {
                id: next_id,
                kind: PackJobKind::ExpandTheme,
                theme_path: new_path,
                depth: parent_job.depth + 1,
                requested_items: parent_job.requested_items,
                safety: subtheme.safety,
            });
        }
    }

    pub fn validate_pack_item(&self, job: &PackJob, item: PackTeacherItem, validator_score: f32) -> ValidatedPackItem {
        let min_confidence = match job.safety {
            SafetyLevel::Normal => self.teacher_settings.min_teacher_confidence,
            SafetyLevel::HighStakes => 0.90,
        };
        let min_validator_score = match job.safety {
            SafetyLevel::Normal => self.teacher_settings.consensus_threshold,
            SafetyLevel::HighStakes => 0.92,
        };
        let accepted = item.confidence >= min_confidence
            && validator_score >= min_validator_score
            && !item.question.trim().is_empty()
            && !item.answer.trim().is_empty();
        ValidatedPackItem {
            theme_path: job.theme_path.clone(),
            question: item.question,
            answer: item.answer,
            teacher_confidence: item.confidence,
            validator_score,
            accepted,
        }
    }

    pub fn train_from_validated_pack_item(&mut self, item: &ValidatedPackItem, input_node: NodeId, output_node: NodeId, forward_passes: usize) -> Option<f32> {
        if !item.accepted {
            self.pack_state.rejected_items += 1;
            return None;
        }
        let input_value = hash_text_to_signal(&item.question);
        let expected_value = hash_text_to_signal(&item.answer);
        let loss = self.train_step(&[(input_node, input_value)], &[(output_node, expected_value)], forward_passes);
        self.pack_state.accepted_items += 1;
        self.pack_state.trained_items += 1;
        Some(loss)
    }


    fn enqueue_data_generation_front(&mut self, job: &PackJob) {
        let new_id = self.next_pack_job_id();
        self.pack_state.queue.push_front(PackJob {
            id: new_id,
            kind: PackJobKind::GenerateTrainingData,
            theme_path: job.theme_path.clone(),
            depth: job.depth,
            requested_items: job.requested_items,
            safety: job.safety,
        });
    }

    fn next_pack_job_id(&self) -> usize {
        self.pack_state.queue.iter().map(|job| job.id).max().unwrap_or(0) + 1
    }

    fn save_path_if_new(&mut self, links: Vec<usize>) -> usize {
        if let Some(existing) = self.paths.iter().find(|p| p.links == links) {
            return existing.id;
        }
        let id = self.paths.len();
        self.paths.push(PreferredPath { id, links, score: 0.0, visits: 0, active: true });
        id
    }

    fn path_would_cycle(&self, path: &[usize], next_link_id: usize) -> bool {
        let next_node = self.correlations[next_link_id].to;
        let mut visited_nodes = HashSet::new();
        for link_id in path {
            visited_nodes.insert(self.correlations[*link_id].from);
            visited_nodes.insert(self.correlations[*link_id].to);
        }
        visited_nodes.contains(&next_node)
    }

    fn link_quality(&self, link_id: usize) -> f32 {
        let link = &self.correlations[link_id];
        link.coefficient.abs() + link.score
    }

    fn path_quality(&self, links: &[usize]) -> f32 {
        if links.is_empty() {
            return 0.0;
        }
        links.iter().map(|id| self.link_quality(*id)).sum::<f32>() / links.len() as f32
    }

    fn reset_temporary_state(&mut self) {
        for grid in &mut self.grids {
            for page in &mut grid.pages {
                for case in &mut page.cases {
                    case.signal = 0.0;
                    case.gradient = 0.0;
                }
            }
        }
        for link in &mut self.correlations {
            link.coefficient_gradient = 0.0;
            link.last_activity = 0.0;
        }
    }

    fn inject_inputs(&mut self, inputs: &[(NodeId, f32)]) {
        for (node, value) in inputs {
            self.add_signal(*node, *value);
        }
    }

    fn assert_valid_node(&self, node: NodeId) {
        assert!(node.grid < self.grids.len(), "grid does not exist: {}", node.grid);
        assert!(node.page < self.grids[node.grid].pages.len(), "page does not exist: grid {}, page {}", node.grid, node.page);
        assert!(node.case < self.grids[node.grid].pages[node.page].cases.len(), "case does not exist: grid {}, page {}, case {}", node.grid, node.page, node.case);
    }

    fn signal(&self, node: NodeId) -> f32 {
        self.grids[node.grid].pages[node.page].cases[node.case].signal
    }

    fn weight(&self, node: NodeId) -> f32 {
        self.grids[node.grid].pages[node.page].cases[node.case].weight
    }

    fn case_gradient(&self, node: NodeId) -> f32 {
        self.grids[node.grid].pages[node.page].cases[node.case].gradient
    }

    fn add_signal(&mut self, node: NodeId, value: f32) {
        self.grids[node.grid].pages[node.page].cases[node.case].signal += value;
    }

    fn add_case_gradient(&mut self, node: NodeId, value: f32) {
        self.grids[node.grid].pages[node.page].cases[node.case].gradient += value;
    }

    fn add_confidence(&mut self, node: NodeId, reward: f32) {
        let case = &mut self.grids[node.grid].pages[node.page].cases[node.case];
        case.confidence = 0.98 * case.confidence + 0.02 * reward;
    }
}

pub fn hash_text_to_signal(text: &str) -> f32 {
    let mut hash: u64 = 14695981039346656037;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    let normalized = (hash % 10_000) as f32 / 10_000.0;
    normalized * 2.0 - 1.0
}

pub fn build_teacher_prompt(job: &PackJob) -> String {
    let theme = job.theme_path.join(" > ");
    let safety_rule = match job.safety {
        SafetyLevel::Normal => "Use only reliable, useful, verifiable, educational knowledge.",
        SafetyLevel::HighStakes => "High-stakes topic. Do not provide personalized advice. Use only general, cautious, verifiable, non-speculative knowledge.",
    };
    format!(
        "Return a compact JSON array only.\n\nTheme: {theme}\nRule: {safety_rule}\n\nSchema exactly:\n[{{\"question\":\"short question\",\"answer\":\"short verified answer\",\"confidence\":0.95,\"tags\":[\"tag\"],\"verification_notes\":\"brief check\"}}]\n\nRules:\n- output must start with [ and end with ]\n- no markdown\n- no code fences\n- no explanation\n- no text before or after JSON\n- answer must be short\n- confidence must be between 0.0 and 1.0\n- maximum {max} item(s)\n",
        theme = theme,
        safety_rule = safety_rule,
        max = job.requested_items
    )
}

pub fn build_subtheme_expansion_prompt(theme_path: &[String], max_children: usize) -> String {
    let path = theme_path.join(" > ");
    format!(
        "Return a compact JSON array only.\n\nCurrent knowledge path: {path}\n\nSchema exactly:\n[{{\"name\":\"Algebra\",\"safety\":\"normal\"}}]\n\nRules:\n- output must start with [ and end with ]\n- no markdown\n- no code fences\n- no explanation\n- no text before or after JSON\n- use short names\n- safety must be normal or high_stakes\n- maximum {max_children} items\n",
        path = path,
        max_children = max_children
    )
}

pub fn universal_root_pack() -> TrainingPack {
    TrainingPack {
        name: "Universal Pack".to_string(),
        enabled: true,
        coverage_mode: PackCoverageMode::Universal,
        max_depth: 6,
        themes: vec![
            root("Sciences"),
            root("Mathematics"),
            root("Computer Science"),
            root("Languages"),
            root("History"),
            root("Geography"),
            root_high("General Health"),
            root_high("General Law"),
            root_high("General Finance"),
            root("Economics"),
            root("Industry"),
            root("Automotive"),
            root("Energy"),
            root("Art"),
            root("Music"),
            root("Cinema"),
            root("Literature"),
            root("Philosophy"),
            root("General Psychology"),
            root("Education"),
            root("Cooking"),
            root("Sports"),
            root("Agriculture"),
            root("Environment"),
            root("Engineering"),
            root("Electronics"),
            root("Robotics"),
            root("Defensive Cybersecurity"),
            root("General Knowledge"),
        ],
    }
}

pub fn default_training_pack() -> TrainingPack {
    TrainingPack {
        name: "General Pack".to_string(),
        enabled: true,
        coverage_mode: PackCoverageMode::Broad,
        max_depth: 3,
        themes: vec![
            ThemeNode { name: "Automotive".to_string(), enabled: true, safety: SafetyLevel::Normal, children: vec![leaf("Mechanics"), leaf("Maintenance"), leaf("Diagnostics"), leaf("Electric Vehicles"), leaf("Road Safety")] },
            ThemeNode { name: "Languages".to_string(), enabled: true, safety: SafetyLevel::Normal, children: vec![leaf("Grammar"), leaf("Vocabulary"), leaf("Translation"), leaf("Idioms"), leaf("Text Correction")] },
            ThemeNode { name: "Finance".to_string(), enabled: true, safety: SafetyLevel::HighStakes, children: vec![leaf_high("Personal Budgeting"), leaf_high("Accounting"), leaf_high("Financial Markets"), leaf_high("General Taxation"), leaf_high("Risk Analysis")] },
            ThemeNode { name: "Code".to_string(), enabled: true, safety: SafetyLevel::Normal, children: vec![leaf("Rust"), leaf("Python"), leaf("Algorithms"), leaf("Software Architecture"), leaf("Debugging"), leaf("Defensive Security")] },
        ],
    }
}

fn root(name: &str) -> ThemeNode {
    ThemeNode { name: name.to_string(), enabled: true, safety: SafetyLevel::Normal, children: vec![] }
}

fn root_high(name: &str) -> ThemeNode {
    ThemeNode { name: name.to_string(), enabled: true, safety: SafetyLevel::HighStakes, children: vec![] }
}

fn leaf(name: &str) -> ThemeNode {
    ThemeNode { name: name.to_string(), enabled: true, safety: SafetyLevel::Normal, children: vec![] }
}

fn leaf_high(name: &str) -> ThemeNode {
    ThemeNode { name: name.to_string(), enabled: true, safety: SafetyLevel::HighStakes, children: vec![] }
}

impl EngravedModel {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

fn dimension_seed(id: usize) -> u64 {
    0xA076_1D64_78BD_642F ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trainer_can_train_and_predict() {
        let mut trainer = RawAITrainer::new(2, 16);
        let input = trainer.official_node(0, 1);
        let output = trainer.official_node(1, 2);
        trainer.add_correlation(input, output, 0.2);
        let loss = trainer.train_step(&[(input, 1.0)], &[(output, 1.0)], 1);
        assert!(loss >= 0.0);
        let prediction = trainer.predict(&[(input, 1.0)], &[output], 1);
        assert_eq!(prediction.len(), 1);
    }

    #[test]
    fn can_create_candidate_page() {
        let mut trainer = RawAITrainer::new(1, 32);
        let official = trainer.grids[0].official_page;
        let zone = trainer.create_critical_zone(0, official, 10, 3);
        assert_eq!(trainer.critical_zones[zone].case_start, 7);
        assert_eq!(trainer.grids[0].pages.len(), 2);
    }

    #[test]
    fn pack_tick_returns_action() {
        let mut trainer = RawAITrainer::new(1, 8);
        trainer.start_pack_training();
        let action = trainer.pack_training_tick();
        assert!(matches!(action, PackTickAction::NeedSubthemeExpansion { .. }));
    }
}

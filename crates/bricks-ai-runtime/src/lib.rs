use bricks_ai_core::{hash_text_to_signal, EngravedModel, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeModel {
    pub model: EngravedModel,
    #[serde(skip)]
    case_weights: HashMap<NodeId, f32>,
    #[serde(skip)]
    adjacency: HashMap<NodeId, Vec<RuntimeCorrelation>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeCorrelation {
    pub from: NodeId,
    pub to: NodeId,
    pub coefficient: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSummary {
    pub cases: usize,
    pub correlations: usize,
    pub input_node: Option<NodeId>,
    pub output_node: Option<NodeId>,
    pub average_case_confidence: f32,
    pub average_correlation_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePrediction {
    pub input_text: String,
    pub input_signal: f32,
    pub output_node: Option<NodeId>,
    pub output_signal: f32,
    pub passes: usize,
    pub activated_nodes: usize,
}

impl RuntimeModel {
    pub fn from_engraved_model(model: EngravedModel) -> Self {
        let mut runtime = Self {
            model,
            case_weights: HashMap::new(),
            adjacency: HashMap::new(),
        };
        runtime.rebuild_indexes();
        runtime
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let text = fs::read_to_string(path)?;
        let model: EngravedModel = serde_json::from_str(&text)?;
        Ok(Self::from_engraved_model(model))
    }

    pub fn save_runtime_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let text = serde_json::to_string_pretty(&self.model)?;
        fs::write(path, text)?;
        Ok(())
    }

    pub fn rebuild_indexes(&mut self) {
        self.case_weights.clear();
        self.adjacency.clear();
        for case in &self.model.cases {
            self.case_weights.insert(NodeId::new(case.grid, case.page, case.case), case.weight);
        }
        for correlation in &self.model.correlations {
            self.adjacency
                .entry(correlation.from)
                .or_default()
                .push(RuntimeCorrelation {
                    from: correlation.from,
                    to: correlation.to,
                    coefficient: correlation.coefficient,
                    score: correlation.score,
                });
        }
    }

    pub fn summary(&self) -> RuntimeSummary {
        let average_case_confidence = if self.model.cases.is_empty() {
            0.0
        } else {
            self.model.cases.iter().map(|case| case.confidence).sum::<f32>() / self.model.cases.len() as f32
        };
        let average_correlation_score = if self.model.correlations.is_empty() {
            0.0
        } else {
            self.model.correlations.iter().map(|corr| corr.score).sum::<f32>() / self.model.correlations.len() as f32
        };
        RuntimeSummary {
            cases: self.model.cases.len(),
            correlations: self.model.correlations.len(),
            input_node: self.default_input_node(),
            output_node: self.default_output_node(),
            average_case_confidence,
            average_correlation_score,
        }
    }

    pub fn default_input_node(&self) -> Option<NodeId> {
        self.model
            .correlations
            .first()
            .map(|corr| corr.from)
            .or_else(|| self.model.cases.first().map(|case| NodeId::new(case.grid, case.page, case.case)))
    }

    pub fn default_output_node(&self) -> Option<NodeId> {
        self.model
            .correlations
            .last()
            .map(|corr| corr.to)
            .or_else(|| self.model.cases.last().map(|case| NodeId::new(case.grid, case.page, case.case)))
    }

    pub fn predict_text(&self, input_text: &str, passes: usize) -> RuntimePrediction {
        let input_signal = hash_text_to_signal(input_text);
        let input_node = self.default_input_node();
        let output_node = self.default_output_node();
        let output_signal = if let (Some(input), Some(output)) = (input_node, output_node) {
            self.propagate(input, input_signal, output, passes)
        } else {
            0.0
        };
        let activated_nodes = self.estimate_activated_nodes(input_node, passes);
        RuntimePrediction {
            input_text: input_text.to_string(),
            input_signal,
            output_node,
            output_signal,
            passes,
            activated_nodes,
        }
    }

    pub fn propagate(&self, input_node: NodeId, input_signal: f32, output_node: NodeId, passes: usize) -> f32 {
        let mut signals: HashMap<NodeId, f32> = HashMap::new();
        signals.insert(input_node, input_signal);
        let safe_passes = passes.max(1);
        for _ in 0..safe_passes {
            let snapshot: Vec<(NodeId, f32)> = signals.iter().map(|(node, signal)| (*node, *signal)).collect();
            for (node, signal) in snapshot {
                if let Some(edges) = self.adjacency.get(&node) {
                    let weight = self.case_weights.get(&node).copied().unwrap_or(1.0);
                    for edge in edges {
                        let influence = signal * weight * edge.coefficient;
                        if influence.abs() > 1e-9 {
                            *signals.entry(edge.to).or_insert(0.0) += influence;
                        }
                    }
                }
            }
        }
        signals.get(&output_node).copied().unwrap_or(0.0)
    }

    fn estimate_activated_nodes(&self, input_node: Option<NodeId>, passes: usize) -> usize {
        let Some(input) = input_node else { return 0; };
        let mut frontier = vec![input];
        let mut visited: HashMap<NodeId, bool> = HashMap::new();
        visited.insert(input, true);
        for _ in 0..passes.max(1) {
            let mut next = Vec::new();
            for node in frontier {
                if let Some(edges) = self.adjacency.get(&node) {
                    for edge in edges {
                        if let std::collections::hash_map::Entry::Vacant(entry) = visited.entry(edge.to) {
                            entry.insert(true);
                            next.push(edge.to);
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        visited.len()
    }
}

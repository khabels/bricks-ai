use bricks_ai_core::{AiProvider, PackStatus, RawAITrainer, TeacherRole};

#[derive(Debug, Clone)]
pub enum UiAction {
    SetProviderEnabled { provider: AiProvider, enabled: bool },
    SetProviderModel { provider: AiProvider, model: String },
    SetProviderRole { provider: AiProvider, role: TeacherRole },
    MarkProviderKeyPresent { provider: AiProvider, present: bool },
    SetConsensusThreshold(f32),
    SetMinimumTeacherConfidence(f32),
    SetCrossValidationRequired(bool),
    StartTraining,
    PauseTraining,
    ResumeTraining,
    SetItemsPerTheme(usize),
    SetMaxDepth(usize),
}

#[derive(Debug, Clone)]
pub struct SettingsPanelSnapshot {
    pub provider_count: usize,
    pub enabled_provider_count: usize,
    pub consensus_threshold: f32,
    pub min_teacher_confidence: f32,
    pub require_cross_validation: bool,
}

#[derive(Debug, Clone)]
pub struct TrainingPanelSnapshot {
    pub status: PackStatus,
    pub queue_len: usize,
    pub accepted_items: usize,
    pub rejected_items: usize,
    pub trained_items: usize,
    pub current_job_path: Option<String>,
}

pub struct UiController;

impl UiController {
    pub fn apply(trainer: &mut RawAITrainer, action: UiAction) {
        match action {
            UiAction::SetProviderEnabled { provider, enabled } => {
                if let Some(config) = trainer.teacher_settings.providers.iter_mut().find(|p| p.provider == provider) {
                    config.enabled = enabled;
                }
            }
            UiAction::SetProviderModel { provider, model } => {
                if let Some(config) = trainer.teacher_settings.providers.iter_mut().find(|p| p.provider == provider) {
                    config.model = model;
                }
            }
            UiAction::SetProviderRole { provider, role } => {
                if let Some(config) = trainer.teacher_settings.providers.iter_mut().find(|p| p.provider == provider) {
                    config.role = role;
                }
            }
            UiAction::MarkProviderKeyPresent { provider, present } => {
                if let Some(config) = trainer.teacher_settings.providers.iter_mut().find(|p| p.provider == provider) {
                    config.api_key_present = present;
                }
            }
            UiAction::SetConsensusThreshold(value) => {
                trainer.teacher_settings.consensus_threshold = value.clamp(0.0, 1.0);
            }
            UiAction::SetMinimumTeacherConfidence(value) => {
                trainer.teacher_settings.min_teacher_confidence = value.clamp(0.0, 1.0);
            }
            UiAction::SetCrossValidationRequired(required) => {
                trainer.teacher_settings.require_cross_validation = required;
            }
            UiAction::StartTraining => trainer.start_pack_training(),
            UiAction::PauseTraining => trainer.pack_state.status = PackStatus::Paused,
            UiAction::ResumeTraining => trainer.pack_state.status = PackStatus::Running,
            UiAction::SetItemsPerTheme(value) => trainer.pack_state.max_items_per_theme = value.max(1),
            UiAction::SetMaxDepth(value) => trainer.training_pack.max_depth = value,
        }
    }

    pub fn settings_snapshot(trainer: &RawAITrainer) -> SettingsPanelSnapshot {
        SettingsPanelSnapshot {
            provider_count: trainer.teacher_settings.providers.len(),
            enabled_provider_count: trainer.teacher_settings.providers.iter().filter(|p| p.enabled).count(),
            consensus_threshold: trainer.teacher_settings.consensus_threshold,
            min_teacher_confidence: trainer.teacher_settings.min_teacher_confidence,
            require_cross_validation: trainer.teacher_settings.require_cross_validation,
        }
    }

    pub fn training_snapshot(trainer: &RawAITrainer) -> TrainingPanelSnapshot {
        TrainingPanelSnapshot {
            status: trainer.pack_state.status,
            queue_len: trainer.pack_state.queue.len(),
            accepted_items: trainer.pack_state.accepted_items,
            rejected_items: trainer.pack_state.rejected_items,
            trained_items: trainer.pack_state.trained_items,
            current_job_path: trainer.pack_state.current_job.as_ref().map(|j| j.theme_path.join(" > ")),
        }
    }
}

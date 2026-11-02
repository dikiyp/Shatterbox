//! wgpu error capture for the autotest harness.
//!
//! The game never aborts on a validation error today: wgpu logs it and the
//! frame continues with whatever the driver decided to do. For automated runs
//! that is not good enough, because a pipeline may have silently fallen back
//! to a built-in shader and the screenshot still looks plausible. This module
//! keeps a process-wide count and the first few messages so the scenario
//! report can fail the run.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static VALIDATION_ERRORS: AtomicU64 = AtomicU64::new(0);
static OUT_OF_MEMORY_ERRORS: AtomicU64 = AtomicU64::new(0);
static INTERNAL_ERRORS: AtomicU64 = AtomicU64::new(0);
static SAMPLES: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

/// How many messages are retained; the report only shows the first five.
const MAX_SAMPLES: usize = 16;

fn samples() -> &'static Mutex<Vec<String>> {
    SAMPLES.get_or_init(|| Mutex::new(Vec::new()))
}

/// Records an uncaptured wgpu error. Safe to call from a device callback.
pub fn record_uncaptured_error(error: &wgpu::Error) {
    let kind = match error {
        wgpu::Error::Validation { .. } => {
            VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
            "validation"
        }
        wgpu::Error::OutOfMemory { .. } => {
            OUT_OF_MEMORY_ERRORS.fetch_add(1, Ordering::Relaxed);
            "out-of-memory"
        }
        wgpu::Error::Internal { .. } => {
            INTERNAL_ERRORS.fetch_add(1, Ordering::Relaxed);
            "internal"
        }
    };
    if let Ok(mut list) = samples().lock() {
        if list.len() < MAX_SAMPLES {
            list.push(format!("[{kind}] {error}"));
        }
    }
    tracing::error!(kind, "{error}");
}

/// Number of validation errors seen since process start.
pub fn validation_error_count() -> u64 {
    VALIDATION_ERRORS.load(Ordering::Relaxed)
}

/// Number of out-of-memory and internal driver errors.
pub fn device_error_count() -> u64 {
    OUT_OF_MEMORY_ERRORS.load(Ordering::Relaxed) + INTERNAL_ERRORS.load(Ordering::Relaxed)
}

/// The first `limit` recorded messages, in arrival order.
pub fn validation_error_samples(limit: usize) -> Vec<String> {
    samples()
        .lock()
        .map(|list| list.iter().take(limit).cloned().collect())
        .unwrap_or_default()
}

/// Installs the capture callback on a device. Called once, right after the
/// device is created and before anything can be submitted.
pub fn install(device: &wgpu::Device) {
    device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
        record_uncaptured_error(&error);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_samples_stay_in_sync() {
        let before = validation_error_count();
        record_uncaptured_error(&wgpu::Error::Validation {
            source: "autotest diagnostic probe".into(),
            description: "autotest diagnostic probe".to_owned(),
        });
        assert_eq!(validation_error_count(), before + 1);
        assert!(
            validation_error_samples(5)
                .iter()
                .any(|sample| sample.contains("autotest diagnostic probe"))
        );
    }
}

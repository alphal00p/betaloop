use crate::{utils::parameterize3d, HardCodedIntegrandSettings, HasIntegrand, Settings};
use num_traits::Inv;
use serde::Deserialize;
use symbolica::numerical_integration::{ContinuousGrid, Grid, Sample};
pub struct Bubble {
    settings: Settings,
    bubble_settings: BubbleSettings,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BubbleSettings {
    m_uv: f64,
    p: [f64; 4],
}

impl HasIntegrand for Bubble {
    fn create_grid(&self) -> Grid<f64> {
        Grid::Continuous(ContinuousGrid::new(
            3,
            self.settings.integrator.n_bins,
            self.settings.integrator.min_samples_for_update,
            None,
            self.settings.integrator.train_on_avg,
        ))
    }

    fn evaluate_sample(
        &mut self,
        sample: &symbolica::numerical_integration::Sample<f64>,
        _wgt: f64,
        _iter: usize,
        use_f128: bool,
    ) -> num::Complex<f64> {
        if use_f128 {
            panic!()
        }

        let sample = match sample {
            Sample::Continuous(_, xs) => xs,
            _ => unreachable!(),
        };

        let (mom, jacobian) =
            parameterize3d(sample, self.settings.kinematics.e_cm, 0, &self.settings);

        let energy_0 = self.energy_0(&mom);
        let energy_1 = self.energy_1(&mom);
        let p0 = self.bubble_settings.p[0];

        let eta_plus = energy_0 + energy_1 + p0;
        let eta_minus = energy_0 + energy_1 - p0;

        let bare = (4. * energy_0 * energy_1).inv() * (eta_plus.inv() + eta_minus.inv());
        let energy_uv = self.energy_uv(&mom);

        let uv_ct = (4. * energy_uv * energy_uv * energy_uv).inv();

        let res = (bare - uv_ct) * jacobian;
        num::Complex::new(res, 0.0)
    }

    fn get_n_dim(&self) -> usize {
        3
    }
}

impl Bubble {
    pub fn new(settings: Settings) -> Self {
        let bubble_settings = match &settings.hard_coded_integrand {
            HardCodedIntegrandSettings::Bubble(bubble_settings) => bubble_settings.clone(),
            _ => unreachable!(),
        };

        Self {
            settings,
            bubble_settings,
        }
    }

    fn energy_0(&self, loop_mom: &[f64; 3]) -> f64 {
        (loop_mom[0] * loop_mom[0] + loop_mom[1] * loop_mom[1] + loop_mom[2] * loop_mom[2]).sqrt()
    }

    fn energy_1(&self, loop_mom: &[f64; 3]) -> f64 {
        let p = &self.bubble_settings.p[1..];
        ((loop_mom[0] + p[0]) * (loop_mom[0] + p[0])
            + (loop_mom[1] + p[1]) * (loop_mom[1] + p[1])
            + (loop_mom[2] + p[2]) * (loop_mom[2] + p[2]))
            .sqrt()
    }

    fn energy_uv(&self, loop_mom: &[f64; 3]) -> f64 {
        (loop_mom[0] * loop_mom[0]
            + loop_mom[1] * loop_mom[1]
            + loop_mom[2] * loop_mom[2]
            + self.bubble_settings.m_uv * self.bubble_settings.m_uv)
            .sqrt()
    }
}

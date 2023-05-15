use crate::integrands::*;
use crate::utils;
use crate::utils::FloatLike;
use crate::Settings;
use crate::{CTVariable, HFunctionSettings};
use colored::Colorize;
use havana::{ContinuousGrid, Grid, Sample};
use lorentz_vector::LorentzVector;
use num::Complex;
use num_traits::ToPrimitive;
use serde::Deserialize;
use utils::{LEFT, MINUS, PLUS, RIGHT};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoopInducedTriBoxTriCTSettings {
    pub variable: CTVariable,
    pub enabled: bool,
    pub h_function: HFunctionSettings,
    pub sliver_width: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoopInducedTriBoxTriSettings {
    pub supegraph_yaml_file: String,
    pub q: [f64; 4],
    pub h_function: HFunctionSettings,
    #[serde(rename = "threshold_CT_settings")]
    pub threshold_ct_settings: LoopInducedTriBoxTriCTSettings,
}

pub struct LoopInducedTriBoxTriIntegrand {
    pub settings: Settings,
    pub supergraph: SuperGraph,
    pub n_dim: usize,
    pub integrand_settings: LoopInducedTriBoxTriSettings,
}
pub struct ComputationCache<T: FloatLike> {
    pub external_momenta: Vec<LorentzVector<T>>,
}

impl<T: FloatLike> ComputationCache<T> {
    pub fn default() -> ComputationCache<T> {
        ComputationCache {
            external_momenta: vec![],
        }
    }
}

#[derive(Debug)]
pub struct ESurfaceCT<T: FloatLike> {
    pub e_surf_id: usize,
    pub ct_basis_signature: Vec<Vec<isize>>,
    pub center_coordinates: Vec<[T; 3]>,
    pub adjusted_sampling_jac: T,
    pub h_function_wgt: T,
    pub e_surf_expanded: T,
    pub scaled_loop_momenta: Vec<LorentzVector<T>>,
    pub onshell_edges: Vec<LorentzVector<T>>,
    pub e_surface_evals: Vec<ESurfaceCache<T>>,
    pub cff_evaluations: Vec<T>,
    pub solution_type: usize,
}

#[derive(Debug)]
pub struct ESurfaceCache<T: FloatLike> {
    pub p1: [T; 3],
    pub p2: [T; 3],
    pub m1sq: T,
    pub m2sq: T,
    pub e_shift: T,
    pub exists: bool,
    pub eval: T,
    pub t_scaling: [T; 2],
}

#[allow(unused)]
impl<T: FloatLike> ESurfaceCache<T> {
    pub fn default() -> ESurfaceCache<T> {
        ESurfaceCache {
            p1: [T::zero(); 3],
            p2: [T::zero(); 3],
            m1sq: T::zero(),
            m2sq: T::zero(),
            e_shift: T::zero(),
            exists: true,
            eval: T::zero(),
            t_scaling: [T::zero(), T::zero()],
        }
    }

    pub fn does_exist(&self) -> bool {
        utils::one_loop_e_surface_exists(&self.p1, &self.p2, self.m1sq, self.m2sq, self.e_shift)
    }

    pub fn bilinear_form(&self) -> ([[T; 3]; 3], [T; 3], T) {
        utils::one_loop_e_surface_bilinear_form(
            &self.p1,
            &self.p2,
            self.m1sq,
            self.m2sq,
            self.e_shift,
        )
    }

    pub fn eval(&self, k: &[T; 3]) -> T {
        utils::one_loop_eval_e_surf(k, &self.p1, &self.p2, self.m1sq, self.m2sq, self.e_shift)
    }

    pub fn eval_t_derivative(&self, k: &[T; 3]) -> T {
        utils::one_loop_eval_e_surf_k_derivative(k, &self.p1, &self.p2, self.m1sq, self.m2sq)
    }

    pub fn compute_t_scaling(&self, k: &[T; 3]) -> [T; 2] {
        utils::one_loop_get_e_surf_t_scaling(
            k,
            &self.p1,
            &self.p2,
            self.m1sq,
            self.m2sq,
            self.e_shift,
        )
    }
}

#[allow(unused)]
impl LoopInducedTriBoxTriIntegrand {
    pub fn new(
        settings: Settings,
        integrand_settings: LoopInducedTriBoxTriSettings,
    ) -> LoopInducedTriBoxTriIntegrand {
        /*
               output_LU_scalar betaLoop_triangleBoxTriangleBenchmark_scalar \
        --topology=[\
           ('q1', 0, 1), \
           ('2', 1, 3), ('7', 3, 4),('4', 4, 5), \
           ('3', 5, 6), ('6', 6, 7),('1', 7, 1), \
           ('5',3,7),('8',4,6),\
           ('q2', 5, 8) ] \
        --masses=[('2', 10.),('1', 10.),('5', 10.),('3', 20.),('4', 20.),('8', 20.),('7', 0.),('6', 0.),] \
        --lmb=('2', '4', '7') \
        --name="betaLoop_triangleBoxTriangleBenchmark_scalar" \
        --analytical_result=0.0e0 \
        --externals=('q1',) \
        --numerator='1'
               */

        // We force the computation to be done with q at rest for now
        assert!(integrand_settings.q[1] == 0.);
        assert!(integrand_settings.q[2] == 0.);
        assert!(integrand_settings.q[3] == 0.);

        let sg = SuperGraph::from_file(integrand_settings.supegraph_yaml_file.as_str()).unwrap();
        let n_dim = utils::get_n_dim_for_n_loop_momenta(&settings, sg.n_loop, false);
        LoopInducedTriBoxTriIntegrand {
            settings,
            supergraph: sg,
            n_dim: n_dim,
            integrand_settings,
        }
    }

    fn evaluate_numerator<T: FloatLike>(
        &self,
        onshell_edge_momenta: &Vec<LorentzVector<T>>,
        left_orientations: &Vec<(usize, isize)>,
        right_orientations: &Vec<(usize, isize)>,
    ) -> T {
        let mut onshell_edge_momenta_flipped = onshell_edge_momenta.clone();
        for (e_id, flip) in left_orientations.iter() {
            onshell_edge_momenta_flipped[*e_id].t *= Into::<T>::into(*flip as f64);
        }
        for (e_id, flip) in right_orientations.iter() {
            onshell_edge_momenta_flipped[*e_id].t *= Into::<T>::into(*flip as f64);
        }

        return T::one();
    }

    fn parameterize<T: FloatLike>(&self, xs: &[T]) -> (Vec<[T; 3]>, T) {
        utils::global_parameterize(
            xs,
            Into::<T>::into(self.settings.kinematics.e_cm * self.settings.kinematics.e_cm),
            &self.settings,
            false,
        )
    }

    fn inv_parameterize<T: FloatLike>(&self, ks: &Vec<LorentzVector<T>>) -> (Vec<T>, T) {
        utils::global_inv_parameterize(
            ks,
            Into::<T>::into(self.settings.kinematics.e_cm * self.settings.kinematics.e_cm),
            &self.settings,
            false,
        )
    }

    fn build_e_surfaces<T: FloatLike>(
        &self,
        onshell_edge_momenta: &Vec<LorentzVector<T>>,
        cache: &ComputationCache<T>,
        amplitude: &Amplitude,
        e_surfaces: &Vec<Esurface>,
        loop_momenta: &Vec<LorentzVector<T>>,
    ) -> Vec<ESurfaceCache<T>> {
        let mut e_surf_caches: Vec<ESurfaceCache<T>> = vec![];

        // Negative edge ids indicate here momenta external to the whole supergraph
        let amplitude_externals = amplitude
            .external_edge_id_and_flip
            .iter()
            .map(|(e_id, flip)| {
                if *e_id < 0 {
                    cache.external_momenta[(-*e_id - 1) as usize] * Into::<T>::into(*flip as f64)
                } else {
                    onshell_edge_momenta[*e_id as usize] * Into::<T>::into(*flip as f64)
                }
            })
            .collect::<Vec<_>>();
        for e_surf in e_surfaces.iter() {
            // Should typically be computed from the signatures, but in this simple case doing like below is easier
            let p1 = onshell_edge_momenta[e_surf.edge_ids[0]]
                - onshell_edge_momenta[amplitude.lmb_edges[0].id];
            let p2 = onshell_edge_momenta[e_surf.edge_ids[1]]
                - onshell_edge_momenta[amplitude.lmb_edges[0].id];
            let mut shift = T::zero();
            for (i_ext, sig) in e_surf.shift.iter().enumerate() {
                shift += amplitude_externals[i_ext].t * Into::<T>::into(*sig as f64);
            }
            let mut e_surf_cache = ESurfaceCache {
                p1: [p1.x, p1.y, p1.z],
                p2: [p2.x, p2.y, p2.z],
                m1sq: Into::<T>::into(
                    self.supergraph.edges[e_surf.edge_ids[0]].mass
                        * self.supergraph.edges[e_surf.edge_ids[0]].mass,
                ),
                m2sq: Into::<T>::into(
                    self.supergraph.edges[e_surf.edge_ids[1]].mass
                        * self.supergraph.edges[e_surf.edge_ids[1]].mass,
                ),
                e_shift: shift,
                exists: true,                      // Will be adjusted later
                eval: T::zero(),                   // Will be adjusted later
                t_scaling: [T::zero(), T::zero()], // Will be adjusted later
            };
            e_surf_cache.exists = e_surf_cache.does_exist();
            let k = onshell_edge_momenta[amplitude.lmb_edges[0].id];
            e_surf_cache.eval = e_surf_cache.eval(&[k.x, k.y, k.z]);
            if e_surf_cache.exists {
                e_surf_cache.t_scaling = e_surf_cache.compute_t_scaling(&[k.x, k.y, k.z]);
            }
            e_surf_caches.push(e_surf_cache);
        }

        e_surf_caches
    }

    fn build_cts_for_e_surf_id_and_amplitude<T: FloatLike>(
        &self,
        side: usize,
        amplitude: &Amplitude,
        e_surf_id: usize,
        t_scaled_loop_momenta: &Vec<LorentzVector<T>>,
        t_scaled_onshell_edge_momenta: &Vec<LorentzVector<T>>,
        cache: &ComputationCache<T>,
        cut: &Cut,
    ) -> Vec<ESurfaceCT<T>> {
        // Quite some gynmastic needs to take place in order to dynamically build the right basis for solving this E-surface CT
        // I hard-code it for now
        let mut ct_basis_signature = if side == LEFT {
            vec![vec![1, 0, 0]]
        } else {
            vec![vec![0, 1, 0]]
        };
        // Same for the center coordinates, the whole convex solver will need to be implemented for this.
        // And of course E-surf multi-channeling on top if there needs to be multiple centers.
        let center_coordinates = vec![[T::zero(); 3]];
        // let center_coordinates = vec![[
        //     Into::<T>::into(0.1),
        //     Into::<T>::into(0.2),
        //     Into::<T>::into(0.3),
        // ]];

        let mut loop_momenta_in_e_surf_basis = t_scaled_loop_momenta.clone();
        if side == LEFT {
            loop_momenta_in_e_surf_basis[0] -= LorentzVector {
                t: T::zero(),
                x: center_coordinates[0][0],
                y: center_coordinates[0][1],
                z: center_coordinates[0][2],
            };
        } else {
            loop_momenta_in_e_surf_basis[1] -= LorentzVector {
                t: T::zero(),
                x: center_coordinates[0][0],
                y: center_coordinates[0][1],
                z: center_coordinates[0][2],
            };
        };
        // The building of the E-surface should be done more generically and efficiently, but here in this simple case we can do it this way
        let subtracted_e_surface = &self.build_e_surfaces(
            &t_scaled_onshell_edge_momenta,
            &cache,
            &amplitude,
            &vec![amplitude.cff_expression.e_surfaces[e_surf_id].clone()],
            &loop_momenta_in_e_surf_basis,
        )[0];

        let center_eval = subtracted_e_surface.eval(&center_coordinates[0]);
        assert!(center_eval < T::zero());
        assert!(subtracted_e_surface.t_scaling[MINUS] < T::zero());
        assert!(subtracted_e_surface.t_scaling[PLUS] > T::zero());

        let t_scaled_r = if side == LEFT {
            t_scaled_loop_momenta[0].spatial_squared().sqrt()
        } else {
            t_scaled_loop_momenta[1].spatial_squared().sqrt()
        };

        let mut all_new_cts = vec![];
        // CHECK: when using R, no negative solution nor absolute value is necessary when using sliver_width <= 1.
        let (solutions_to_consider, sliver_width) = match self
            .integrand_settings
            .threshold_ct_settings
            .variable
        {
            // This would be for hemispherical coordinates
            // CTVariable::Radius => vec![PLUS, MINUS],
            CTVariable::Radius => (
                vec![PLUS],
                if let Some(s) = self.integrand_settings.threshold_ct_settings.sliver_width {
                    if s > 1. {
                        panic!("Solving threshold CTs in the R variable is not yet implemented for sliver_width > 1 as this required hemispherical coordinates.");
                    } else {
                        Some(Into::<T>::into(s as f64))
                    }
                } else {
                    Some(T::one())
                },
            ),
            CTVariable::LogRadius => (
                vec![PLUS],
                self.integrand_settings
                    .threshold_ct_settings
                    .sliver_width
                    .map(|s| Into::<T>::into(s as f64)),
            ),
        };
        for solution_type in solutions_to_consider {
            let scaled_loop_momentum_e_surf_basis = if side == LEFT {
                loop_momenta_in_e_surf_basis[0] * subtracted_e_surface.t_scaling[solution_type]
            } else {
                loop_momenta_in_e_surf_basis[1] * subtracted_e_surface.t_scaling[solution_type]
            };

            let r_star = scaled_loop_momentum_e_surf_basis
                .spatial_squared()
                .abs()
                .sqrt();
            let r = r_star / subtracted_e_surface.t_scaling[solution_type];

            let (mut t, mut t_star) = match self.integrand_settings.threshold_ct_settings.variable {
                CTVariable::Radius => (r, r_star),
                CTVariable::LogRadius => (r.ln(), r_star.ln()),
            };

            if let Some(sliver) = sliver_width {
                if ((t - t_star) / t_star) * ((t - t_star) / t_star) > sliver * sliver {
                    continue;
                }
            }

            let mut ct_scaled_loop_momenta = loop_momenta_in_e_surf_basis.clone();
            if side == LEFT {
                ct_scaled_loop_momenta[0] = scaled_loop_momentum_e_surf_basis;
            } else {
                ct_scaled_loop_momenta[1] = scaled_loop_momentum_e_surf_basis;
            }

            let onshell_edge_momenta_for_this_ct = self.evaluate_onshell_edge_momenta(
                &ct_scaled_loop_momenta,
                &cache.external_momenta,
                cut,
            );
            let e_surface_cache_for_this_ct = self.build_e_surfaces(
                &onshell_edge_momenta_for_this_ct,
                &cache,
                &amplitude,
                &amplitude.cff_expression.e_surfaces,
                &ct_scaled_loop_momenta,
            );

            let mut e_surf_expanded = subtracted_e_surface.eval_t_derivative(&[
                scaled_loop_momentum_e_surf_basis.x,
                scaled_loop_momentum_e_surf_basis.y,
                scaled_loop_momentum_e_surf_basis.z,
            ]) * (t - t_star);
            match self.integrand_settings.threshold_ct_settings.variable {
                CTVariable::Radius => {
                    e_surf_expanded *= r_star.inv();
                }
                CTVariable::LogRadius => {}
            }

            let h_function_wgt = utils::h(
                t,
                Some(t_star),
                None,
                &self.integrand_settings.threshold_ct_settings.h_function,
            );

            // This step is not necessary here because we have the same solving center than sampling one
            // but in general we have to do it when the CT center is different
            // We downscale back the loop momenta to the original scale because we only want to capture the
            // difference in the Jacobian coming from the change of basis and center, not of hyperadius.
            let mut ct_rescaled_loop_momenta = ct_scaled_loop_momenta.clone();
            if side == LEFT {
                let rescaling_factor = t_scaled_loop_momenta[0].spatial_squared().sqrt()
                    / ct_rescaled_loop_momenta[0].spatial_squared().sqrt();
                ct_rescaled_loop_momenta[0] *= rescaling_factor;
            } else {
                let rescaling_factor = t_scaled_loop_momenta[1].spatial_squared().sqrt()
                    / ct_rescaled_loop_momenta[1].spatial_squared().sqrt();
                ct_rescaled_loop_momenta[1] *= rescaling_factor;
            }
            let (_xs, inv_param_jac) = self.inv_parameterize(&ct_rescaled_loop_momenta);
            let (_xs, inv_param_jac_orig) = self.inv_parameterize(&t_scaled_loop_momenta);
            let param_jac = inv_param_jac.inv();
            let param_jac_orig = inv_param_jac_orig.inv();

            // TODO: INVESTIGATE WHAT TO DO IF THE RATIO BELOW IS NOT ONE!!
            let mut adjusted_sampling_jac = param_jac / param_jac_orig;
            // Now account for the radius impact on the jacobian. TOCHECK: IS THIS FACTOR THEN UNIVERSAL (THE SAME FOR ANY SAMPLING PARAMETERISATION) ONCE THE ABOVE FACTOR IS CONSIDERED?
            adjusted_sampling_jac *= (r_star / t_scaled_r).powi(3 - 1);
            //println!("adjusted_sampling_jac = {}", adjusted_sampling_jac);

            match self.integrand_settings.threshold_ct_settings.variable {
                CTVariable::Radius => {}
                CTVariable::LogRadius => {
                    adjusted_sampling_jac *= r_star / t_scaled_r;
                }
            }

            let mut cff_evaluations = vec![];
            for (i_cff, cff_term) in amplitude.cff_expression.terms.iter().enumerate() {
                if !cff_term.contains_e_surf_id(e_surf_id) {
                    cff_evaluations.push(T::zero());
                } else {
                    cff_evaluations.push(cff_term.evaluate(
                        &e_surface_cache_for_this_ct,
                        Some((e_surf_id, e_surf_expanded)),
                    ));
                }
            }

            all_new_cts.push(ESurfaceCT {
                e_surf_id,
                ct_basis_signature: ct_basis_signature.clone(),
                center_coordinates: center_coordinates.clone(),
                adjusted_sampling_jac,
                h_function_wgt,
                e_surf_expanded,
                scaled_loop_momenta: ct_scaled_loop_momenta,
                onshell_edges: onshell_edge_momenta_for_this_ct,
                e_surface_evals: e_surface_cache_for_this_ct,
                solution_type,
                cff_evaluations,
            });
        }

        all_new_cts
    }

    fn evaluate_onshell_edge_momenta<T: FloatLike>(
        &self,
        loop_momenta: &Vec<LorentzVector<T>>,
        external_momenta: &Vec<LorentzVector<T>>,
        cut: &Cut,
    ) -> Vec<LorentzVector<T>> {
        let mut onshell_edge_momenta = vec![];
        for (i, e) in self.supergraph.edges.iter().enumerate() {
            let mut edge_mom =
                utils::compute_momentum(&e.signature, &loop_momenta, &external_momenta);
            edge_mom.t = (edge_mom.spatial_squared()
                + Into::<T>::into(e.mass) * Into::<T>::into(e.mass))
            .sqrt();
            onshell_edge_momenta.push(edge_mom);
        }
        for (e_id, flip) in cut.cut_edge_ids_and_flip.iter() {
            onshell_edge_momenta[*e_id].t *= Into::<T>::into(*flip as f64);
        }
        onshell_edge_momenta
    }

    fn evaluate_cut<T: FloatLike>(
        &self,
        i_cut: usize,
        cut: &Cut,
        loop_momenta: &Vec<LorentzVector<T>>,
        overall_sampling_jac: T,
        cache: &ComputationCache<T>,
    ) -> Complex<T> {
        let mut cut_res = Complex::new(T::zero(), T::zero());

        if self.settings.general.debug > 1 {
            println!(
                "{}",
                format!(
                "  > Starting evaluation of cut #{} ( n_loop_left={} | cut_cardinality={} | n_loop_right={} )",
                i_cut,
                cut.left_amplitude.n_loop,
                cut.cut_edge_ids_and_flip.len(),
                cut.right_amplitude.n_loop,            )
                .blue()
            );
        }

        // Include constants and flux factor
        let mut constants = Complex::new(T::one(), T::zero());
        constants /= Into::<T>::into(2 as f64) * cache.external_momenta[0].square().sqrt().abs();
        // And the 2 pi for each edge
        constants /= (Into::<T>::into(2 as f64) * T::PI()).powi(self.supergraph.edges.len() as i32);

        // Evaluate kinematics before forcing correct hyperradius
        let onshell_edge_momenta_for_this_cut =
            self.evaluate_onshell_edge_momenta(&loop_momenta, &cache.external_momenta, cut);

        // Build the E-surface corresponding to this Cutkosky cut, make sure our hard-coded assumption for this cut holds
        assert!(
            self.supergraph.edges[cut.cut_edge_ids_and_flip[0].0]
                .signature
                .0
                == vec![0, 0, 1]
        );
        assert!(
            self.supergraph.edges[cut.cut_edge_ids_and_flip[1].0]
                .signature
                .0
                == vec![0, 0, 1]
        );

        // We identify the shifts by subtracting the loop momentum contribution, in general it'd be better to to this from the signature directly
        let p1 =
            onshell_edge_momenta_for_this_cut[cut.cut_edge_ids_and_flip[0].0] - loop_momenta[2];
        let p2 =
            onshell_edge_momenta_for_this_cut[cut.cut_edge_ids_and_flip[1].0] - loop_momenta[2];
        let mut e_surface_cc_cut = ESurfaceCache {
            p1: [p1.x, p1.y, p1.z],
            p2: [p2.x, p2.y, p2.z],
            m1sq: Into::<T>::into(self.supergraph.edges[cut.cut_edge_ids_and_flip[0].0].mass)
                * Into::<T>::into(self.supergraph.edges[cut.cut_edge_ids_and_flip[0].0].mass),
            m2sq: Into::<T>::into(self.supergraph.edges[cut.cut_edge_ids_and_flip[1].0].mass)
                * Into::<T>::into(self.supergraph.edges[cut.cut_edge_ids_and_flip[1].0].mass),
            e_shift: -Into::<T>::into(self.integrand_settings.q[0]),
            eval: T::zero(),
            exists: true,
            t_scaling: [T::zero(), T::zero()],
        };
        //println!("e_surface_cc_cut={:?}", e_surface_cc_cut);
        e_surface_cc_cut.t_scaling = e_surface_cc_cut.compute_t_scaling(&[
            loop_momenta[2].x,
            loop_momenta[2].y,
            loop_momenta[2].z,
        ]);

        let rescaled_loop_momenta = vec![
            loop_momenta[0] * e_surface_cc_cut.t_scaling[0],
            loop_momenta[1] * e_surface_cc_cut.t_scaling[0],
            loop_momenta[2] * e_surface_cc_cut.t_scaling[0],
        ];

        // Compute the jacobian of the rescaling
        let normalised_hyperradius = (rescaled_loop_momenta[0].spatial_squared()
            + rescaled_loop_momenta[1].spatial_squared()
            + rescaled_loop_momenta[2].spatial_squared())
        .sqrt()
            / Into::<T>::into(
                self.settings.kinematics.e_cm * self.integrand_settings.h_function.sigma,
            );

        // Very interesting: note that setting sigma to be a function of the other variables works!
        // This is because one can always imagine doing the t-integral last! All that matters is that at t_star dual cancelations remain!
        let cut_h_function = utils::h(
            e_surface_cc_cut.t_scaling[0],
            None,
            None, // Some(normalised_hyperradius.powi(2))
            &self.integrand_settings.h_function,
        );
        let mut t_scaling_jacobian =
            e_surface_cc_cut.t_scaling[0].powi((3 * self.supergraph.n_loop) as i32);

        if self.settings.general.debug > 1 {
            println!(
                "    Rescaling for this cut: {:+e}",
                e_surface_cc_cut.t_scaling[0]
            );
            println!(
                "    Normalised hyperradius for this cut: {:+e}",
                normalised_hyperradius
            );
            println!(
                "    t-scaling jacobian and h-function for this cut: {:+e}",
                t_scaling_jacobian * cut_h_function
            );
        }

        // Include the t-derivative of the E-surface as well
        let cut_e_surface_derivative = e_surface_cc_cut.t_scaling[0]
            / e_surface_cc_cut.eval_t_derivative(&[
                rescaled_loop_momenta[2].x,
                rescaled_loop_momenta[2].y,
                rescaled_loop_momenta[2].z,
            ]);

        // Now re-evaluate the kinematics with the correct hyperradius
        let onshell_edge_momenta_for_this_cut = self.evaluate_onshell_edge_momenta(
            &rescaled_loop_momenta,
            &cache.external_momenta,
            cut,
        );

        if self.settings.general.debug > 1 {
            println!("    Edge on-shell momenta for this cut:");
            for (i, l) in onshell_edge_momenta_for_this_cut.iter().enumerate() {
                println!(
                    "      q{} = ( {:-40}, {:-40}, {:-40}, {:-40} )",
                    i,
                    format!("{:+e}", l.t),
                    format!("{:+e}", l.x),
                    format!("{:+e}", l.y),
                    format!("{:+e}", l.z)
                );
            }
        }

        // Evaluate E-surfaces
        let mut e_surf_caches: [Vec<ESurfaceCache<T>>; 2] = [vec![], vec![]];
        for side in [LEFT, RIGHT] {
            let amplitude = if side == LEFT {
                &cut.left_amplitude
            } else {
                &cut.right_amplitude
            };
            e_surf_caches[side] = self.build_e_surfaces(
                &onshell_edge_momenta_for_this_cut,
                &cache,
                &amplitude,
                &amplitude.cff_expression.e_surfaces,
                &rescaled_loop_momenta,
            );
        }

        let mut i_term = 0;
        let mut cff_evaluations = [vec![], vec![]];
        for side in 0..=1 {
            let amplitude = if side == LEFT {
                &cut.left_amplitude
            } else {
                &cut.right_amplitude
            };
            for (i_cff, cff_term) in amplitude.cff_expression.terms.iter().enumerate() {
                let cff_eval = cff_term.evaluate(&e_surf_caches[side], None);
                if self.settings.general.debug > 2 {
                    println!(
                        "   > {} cFF evaluation for orientation #{}({}): {}",
                        if side == LEFT { "Left" } else { "Right" },
                        format!("{}", i_cff).green(),
                        cff_term
                            .orientation
                            .iter()
                            .map(|(_id, flip)| if *flip > 0 { "+" } else { "-" })
                            .collect::<Vec<_>>()
                            .join("")
                            .blue(),
                        format!("{:+e}", cff_eval).blue()
                    );
                }
                cff_evaluations[side].push(cff_eval);
            }
        }

        let mut e_product_left = T::one();
        for e in cut.left_amplitude.edges.iter() {
            e_product_left *=
                Into::<T>::into(2 as f64) * onshell_edge_momenta_for_this_cut[e.id].t.abs();
        }
        let mut e_product_right = T::one();
        for e in cut.right_amplitude.edges.iter() {
            e_product_right *=
                Into::<T>::into(2 as f64) * onshell_edge_momenta_for_this_cut[e.id].t.abs();
        }

        // Now build the counterterms
        let mut cts = [vec![], vec![]];
        if self.integrand_settings.threshold_ct_settings.enabled {
            // There are smarter ways to do this, but this is the most straightforward and clear for this exploration
            for side in [LEFT, RIGHT] {
                let amplitude = if side == LEFT {
                    &cut.left_amplitude
                } else {
                    &cut.right_amplitude
                };

                for (e_surf_id, _e_surface) in
                    amplitude.cff_expression.e_surfaces.iter().enumerate()
                {
                    if e_surf_caches[side][e_surf_id].exists {
                        cts[side].extend(self.build_cts_for_e_surf_id_and_amplitude(
                            side,
                            amplitude,
                            e_surf_id,
                            &rescaled_loop_momenta,
                            &onshell_edge_momenta_for_this_cut,
                            cache,
                            cut,
                        ))
                    }
                }
            }
        }

        if self.settings.general.debug > 2 {
            println!(
                "   > Number of active CTs on either each side of the cut: left={}, right={}:",
                format!("{}", cts[LEFT].len()).blue(),
                format!("{}", cts[RIGHT].len()).blue(),
            );
        }

        // Note that we could also consider splitting the numerator into a left and right component, depending on its implementation
        let mut cff_sum = Complex::new(T::zero(), T::zero());
        let mut cff_cts_sum = Complex::new(T::zero(), T::zero());
        for (left_i_cff, left_cff_term) in
            cut.left_amplitude.cff_expression.terms.iter().enumerate()
        {
            for (right_i_cff, right_cff_term) in
                cut.right_amplitude.cff_expression.terms.iter().enumerate()
            {
                i_term += 1;
                let mut numerator_wgt = self.evaluate_numerator(
                    &onshell_edge_momenta_for_this_cut,
                    &left_cff_term.orientation,
                    &right_cff_term.orientation,
                );

                let cff_left_wgt = cff_evaluations[0][left_i_cff];
                let cff_right_wgt = cff_evaluations[1][right_i_cff];

                cff_sum += numerator_wgt
                    * cff_left_wgt
                    * e_product_left.inv()
                    * cff_right_wgt
                    * e_product_right.inv();

                // Now include counterterms
                let amplitudes_pair = [&cut.left_amplitude, &cut.right_amplitude];
                let i_cff_pair = [left_i_cff, right_i_cff];
                let mut cts_sum_for_this_term = T::zero();
                for ct_side in [LEFT, RIGHT] {
                    for ct in cts[ct_side].iter() {
                        if ct.cff_evaluations[i_cff_pair[ct_side]] == T::zero() {
                            continue;
                        }
                        let ct_numerator_wgt = self.evaluate_numerator(
                            &ct.onshell_edges,
                            &left_cff_term.orientation,
                            &right_cff_term.orientation,
                        );
                        let mut ct_e_product = T::one();
                        for e in amplitudes_pair[ct_side].edges.iter() {
                            ct_e_product *=
                                Into::<T>::into(2 as f64) * ct.onshell_edges[e.id].t.abs();
                        }
                        let other_side_terms = if ct_side == LEFT {
                            cff_right_wgt * e_product_right.inv()
                        } else {
                            cff_left_wgt * e_product_left.inv()
                        };

                        let ct_weight = -other_side_terms
                            * ct.adjusted_sampling_jac
                            * ct_numerator_wgt
                            * ct.cff_evaluations[i_cff_pair[ct_side]]
                            * ct_e_product.inv()
                            * ct.h_function_wgt;
                        // println!("other_side_terms = {:+e}", other_side_terms);
                        // println!("ct.adjusted_sampling_jac = {:+e}", ct.adjusted_sampling_jac);
                        // println!("ct_numerator_wgt = {:+e}", ct_numerator_wgt);
                        // println!(
                        //     "ct.cff_evaluations[i_cff_pair[ct_side]] = {:+e}",
                        //     ct.cff_evaluations[i_cff_pair[ct_side]]
                        // );
                        // println!("ct type = {}", ct.solution_type);
                        // println!("ct_e_product = {:+e}", ct_e_product);
                        // println!("e_product_left = {:+e}", e_product_left);
                        // println!("ct.h_function_wgt = {:+e}", ct.h_function_wgt);
                        // println!("ct.e_surf_derivative_wgt = {:+e}", ct.e_surf_expanded);
                        // println!(
                        //     "A = {} vs B = {}, A/B = {}",
                        //     cff_left_wgt,
                        //     ct.cff_evaluations[i_cff_pair[ct_side]],
                        //     cff_left_wgt / ct.cff_evaluations[i_cff_pair[ct_side]],
                        // );
                        if self.settings.general.debug > 3 {
                            println!(
                                "   > cFF Evaluation #{} : CT for {} E-surface #{} : {:+e}",
                                format!("{}", i_term).green(),
                                if ct_side == LEFT {
                                    format!("{}", "left").purple()
                                } else {
                                    format!("{}", "right").purple()
                                },
                                ct.e_surf_id,
                                ct_weight
                            );
                        }
                        cts_sum_for_this_term += ct_weight;
                    }
                }
                cff_cts_sum += cts_sum_for_this_term;

                if self.settings.general.debug > 2 {
                    println!(
                        "   > cFF evaluation #{} for orientation #{}({}) x #{}({}):",
                        format!("{}", i_term).green(),
                        format!("{}", left_i_cff).green(),
                        left_cff_term
                            .orientation
                            .iter()
                            .map(|(_id, flip)| if *flip > 0 { "+" } else { "-" })
                            .collect::<Vec<_>>()
                            .join("")
                            .blue(),
                        format!("{}", right_i_cff).green(),
                        right_cff_term
                            .orientation
                            .iter()
                            .map(|(_id, flip)| if *flip > 0 { "+" } else { "-" })
                            .collect::<Vec<_>>()
                            .join("")
                            .blue(),
                    );
                    println!("     left  : {:+e}", cff_left_wgt * e_product_left.inv());
                    println!("     right : {:+e}", cff_right_wgt * e_product_right.inv());
                    println!("     num   : {:+e}", numerator_wgt);
                    println!(
                        "{}",
                        format!(
                            "     tot   : {:+e}",
                            numerator_wgt
                                * cff_left_wgt
                                * e_product_left.inv()
                                * cff_right_wgt
                                * e_product_right.inv()
                        )
                        .green()
                        .bold()
                    );
                    println!(
                        "{}",
                        format!("     ∑ CTs : {:+.e}", cts_sum_for_this_term).green()
                    );
                }
            }
        }

        let mut e_product_cut = T::one();
        for (e_id, _flip) in cut.cut_edge_ids_and_flip.iter() {
            e_product_cut *=
                Into::<T>::into(2 as f64) * onshell_edge_momenta_for_this_cut[*e_id].t.abs();
        }
        if self.settings.general.debug > 1 {
            println!(
                "  > On-shell energy product: {:+e} x {:+e} x {:+e} = {:+e}",
                e_product_left,
                e_product_cut,
                e_product_right,
                e_product_left * e_product_cut * e_product_right
            );
        }
        // Collect terms
        cut_res = cff_sum + cff_cts_sum;

        // Collect all factors that are common for the original integrand and the threshold counterterms
        cut_res *= constants;
        cut_res *= overall_sampling_jac;
        cut_res *= t_scaling_jacobian;
        cut_res *= cut_h_function;
        cut_res *= cut_e_surface_derivative;
        cut_res *= e_product_cut.inv();

        if self.settings.general.debug > 1 {
            println!(
            "{}",
            format!(
            "  > Result for cut #{} ( n_loop_left={} | cut_cardinality={} | n_loop_right={} ): {:+e} ( ∑ CTs = {:+e} )",
            i_cut,
            cut.left_amplitude.n_loop,
            cut.cut_edge_ids_and_flip.len(),
            cut.right_amplitude.n_loop,
            cut_res,
            cff_cts_sum * constants * overall_sampling_jac * t_scaling_jacobian * cut_h_function * cut_e_surface_derivative * e_product_cut.inv()
        )
            .green()
        );
        }

        return cut_res;
    }

    fn evaluate_sample_generic<T: FloatLike>(&self, xs: &[T]) -> Complex<T> {
        let (moms, overall_sampling_jac) = self.parameterize(xs);

        let mut loop_momenta = vec![];
        for m in &moms {
            loop_momenta.push(LorentzVector::from_args(T::zero(), m[0], m[1], m[2]));
        }
        if self.settings.general.debug > 1 {
            println!(
                "Sampled loop momenta (using e_cm = {:.16}):",
                self.settings.kinematics.e_cm
            );
            for (i, l) in loop_momenta.iter().enumerate() {
                println!(
                    "k{} = ( {:-40}, {:-40}, {:-40}, {:-40} )",
                    i,
                    format!("{:+e}", l.t),
                    format!("{:+e}", l.x),
                    format!("{:+e}", l.y),
                    format!("{:+e}", l.z)
                );
            }
        }

        let mut computational_cache = ComputationCache::default();

        for i in 0..=1 {
            computational_cache
                .external_momenta
                .push(LorentzVector::from_args(
                    Into::<T>::into(self.integrand_settings.q[0]),
                    Into::<T>::into(self.integrand_settings.q[1]),
                    Into::<T>::into(self.integrand_settings.q[2]),
                    Into::<T>::into(self.integrand_settings.q[3]),
                ));
        }

        let mut final_wgt = Complex::new(T::zero(), T::zero());
        for (i_cut, cut) in self.supergraph.cuts.iter().enumerate() {
            final_wgt += self.evaluate_cut(
                i_cut,
                cut,
                &loop_momenta,
                overall_sampling_jac,
                &computational_cache,
            );
        }

        if self.settings.general.debug > 0 {
            println!(
                "{}",
                format!("total cuts weight : {:+e}", final_wgt)
                    .green()
                    .bold()
            );
            println!(
                "{}",
                format!("Sampling jacobian : {:+e}", overall_sampling_jac)
                    .green()
                    .bold()
            );
            println!(
                "{}",
                format!("Final contribution: {:+e}", final_wgt)
                    .green()
                    .bold()
            );
        }
        return final_wgt;
    }
}

#[allow(unused)]
impl HasIntegrand for LoopInducedTriBoxTriIntegrand {
    fn create_grid(&self) -> Grid {
        Grid::ContinuousGrid(ContinuousGrid::new(
            self.n_dim,
            self.settings.integrator.n_bins,
            self.settings.integrator.min_samples_for_update,
        ))
    }

    fn get_n_dim(&self) -> usize {
        return self.n_dim;
    }

    fn evaluate_sample(
        &self,
        sample: &Sample,
        wgt: f64,
        iter: usize,
        use_f128: bool,
    ) -> Complex<f64> {
        let xs = match sample {
            Sample::ContinuousGrid(_w, v) => v,
            _ => panic!("Wrong sample type"),
        };

        let mut sample_xs = vec![];
        sample_xs.extend(xs);
        if self.settings.general.debug > 1 {
            println!(
                "Sampled x-space : ( {} )",
                sample_xs
                    .iter()
                    .map(|&x| format!("{:.16}", x))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!("Integrator weight : {:+.16e}", wgt);
        }

        // TODO implement stability check

        if use_f128 {
            let sample_xs_f128 = sample_xs
                .iter()
                .map(|x| Into::<f128::f128>::into(*x))
                .collect::<Vec<_>>();
            if self.settings.general.debug > 1 {
                println!(
                    "f128 Upcasted x-space sample : ( {} )",
                    sample_xs_f128
                        .iter()
                        .map(|&x| format!("{:+e}", x))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            let res = self.evaluate_sample_generic(sample_xs_f128.as_slice());
            return Complex::new(
                f128::f128::to_f64(&res.re).unwrap(),
                f128::f128::to_f64(&res.im).unwrap(),
            );
        } else {
            return self.evaluate_sample_generic(sample_xs.as_slice());
        }
    }
}

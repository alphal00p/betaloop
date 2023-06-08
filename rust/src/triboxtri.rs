use crate::integrands::*;
#[allow(unused)]
use crate::observables::{EventManager, Observables};
use crate::utils;
use crate::utils::FloatLike;
use crate::Settings;
use crate::{CTVariable, HFunctionSettings, NumeratorType};
use colored::Colorize;
use havana::{ContinuousGrid, Grid, Sample};
use lorentz_vector::LorentzVector;
use num::Complex;
use num_traits::FloatConst;
use num_traits::ToPrimitive;
use serde::Deserialize;
use utils::{AMPLITUDE_LEVEL_CT, LEFT, MINUS, PLUS, RIGHT, SUPERGRAPH_LEVEL_CT};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TriBoxTriCTSettings {
    pub variable: CTVariable,
    pub enabled: bool,
    pub compute_only_im_squared: bool,
    pub im_squared_through_local_ct_only: bool,
    pub include_integrated_ct: bool,
    pub include_amplitude_level_cts: bool,
    pub parameterization_center: Vec<[f64; 3]>,
    pub local_ct_h_function: HFunctionSettings,
    pub integrated_ct_h_function: HFunctionSettings,
    pub local_ct_sliver_width: Option<f64>,
    pub integrated_ct_sliver_width: Option<f64>,
    pub apply_anti_observable: bool,
    pub pinch_dampening: PinchDampeningSettings,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PinchDampeningSettings {
    pub enabled: bool,
    pub global: bool,
    pub dampen_all_pinch_surfaces: bool,
    pub regularization_multiplier: f64,
    pub powers: (i32, i32),
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TriBoxTriSettings {
    pub supergraph_yaml_file: String,
    pub q: [f64; 4],
    pub h_function: HFunctionSettings,
    pub numerator: NumeratorType,
    pub sampling_basis: Vec<usize>,
    #[serde(rename = "threshold_CT_settings")]
    pub threshold_ct_settings: TriBoxTriCTSettings,
}

pub struct TriBoxTriIntegrand {
    pub settings: Settings,
    pub supergraph: SuperGraph,
    pub n_dim: usize,
    pub integrand_settings: TriBoxTriSettings,
    pub event_manager: EventManager,
    pub sampling_rot: Option<[[isize; 3]; 3]>,
}

pub fn e_surf_str(e_surf_id: usize, e_surf: &Esurface) -> String {
    format!(
        "#{:<3} edge_ids={:-8} e_shift={:-10}",
        e_surf_id,
        format!(
            "[{}]",
            e_surf
                .edge_ids
                .iter()
                .map(|e_id| format!("{}", *e_id))
                .collect::<Vec<_>>()
                .join(",")
        ),
        format!(
            "[{}]",
            e_surf
                .shift
                .iter()
                .map(|s| format!("{}", *s))
                .collect::<Vec<_>>()
                .join(",")
        )
    )
}

#[derive(Debug, Clone)]
struct ClosestESurfaceMonitor<T: FloatLike + std::fmt::Debug> {
    distance: T,
    i_cut: usize,
    side: usize,
    e_surf_id: usize,
    e_surf: Esurface,
    e_surf_cache: GenericESurfaceCache<T>,
}

impl<T: FloatLike + std::fmt::Debug> ClosestESurfaceMonitor<T> {
    pub fn str_form(&self, cut: &Cut) -> String {
        format!(
            "Normalised distance: {:+e} | E-surface {} of cut #{}{}: {}",
            self.distance,
            if self.side == LEFT { "left" } else { "right" },
            self.i_cut,
            format!(
                "({})",
                cut.cut_edge_ids_and_flip
                    .iter()
                    .map(|(id, flip)| if *flip > 0 {
                        format!("+{}", id)
                    } else {
                        format!("-{}", id)
                    })
                    .collect::<Vec<_>>()
                    .join("|")
            ),
            e_surf_str(self.e_surf_id, &self.e_surf)
        )
    }
}

pub fn compute_propagator_momentum<T: FloatLike>(
    prop_signature: &Vec<(isize, isize)>,
    onshell_edge_momenta: &Vec<LorentzVector<T>>,
    cache: &ComputationCache<T>,
) -> LorentzVector<T> {
    let mut prop_momentum = LorentzVector {
        t: T::zero(),
        x: T::zero(),
        y: T::zero(),
        z: T::zero(),
    };
    for (e_id, flip) in prop_signature {
        prop_momentum += if *e_id < 0 {
            cache.external_momenta[(-*e_id - 1) as usize] * Into::<T>::into(*flip as f64)
        } else {
            onshell_edge_momenta[*e_id as usize] * Into::<T>::into(*flip as f64)
        };
    }
    prop_momentum
}

#[allow(unused)]
impl TriBoxTriIntegrand {
    pub fn new(settings: Settings, integrand_settings: TriBoxTriSettings) -> TriBoxTriIntegrand {
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

        let sg = SuperGraph::from_file(integrand_settings.supergraph_yaml_file.as_str()).unwrap();
        let n_dim = utils::get_n_dim_for_n_loop_momenta(&settings, sg.n_loop, false);
        let event_manager = EventManager::new(true, settings.clone());
        TriBoxTriIntegrand {
            settings,
            supergraph: sg,
            n_dim: n_dim,
            integrand_settings,
            event_manager,
            sampling_rot: None,
        }
    }

    fn evaluate_numerator<T: FloatLike>(
        &self,
        onshell_edge_momenta: &Vec<LorentzVector<T>>,
        left_orientations: &Vec<(usize, isize)>,
        right_orientations: &Vec<(usize, isize)>,
    ) -> T {
        match self.integrand_settings.numerator {
            NumeratorType::One => {
                return T::one();
            }
            NumeratorType::SpatialExponentialDummy => {
                let num_arg = (onshell_edge_momenta[1].spatial_squared()
                    + onshell_edge_momenta[3].spatial_squared()
                    + onshell_edge_momenta[6].spatial_squared())
                    / (Into::<T>::into((self.settings.kinematics.e_cm * 1.).powi(2) as f64));
                return (-num_arg).exp();
            }
            NumeratorType::Physical => {
                let mut onshell_edge_momenta_flipped = onshell_edge_momenta.clone();
                for (e_id, flip) in left_orientations.iter() {
                    onshell_edge_momenta_flipped[*e_id].t *= Into::<T>::into(*flip as f64);
                }
                for (e_id, flip) in right_orientations.iter() {
                    onshell_edge_momenta_flipped[*e_id].t *= Into::<T>::into(*flip as f64);
                }
                unimplemented!()
            }
        }
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

    fn loop_indices_in_edge_ids(&self, edge_ids: &Vec<usize>) -> Vec<usize> {
        let mut edge_indices = vec![];
        for e_id in edge_ids.iter() {
            for (i_sig, s) in self.supergraph.edges[*e_id].signature.0.iter().enumerate() {
                if *s != 0 && !edge_indices.contains(&i_sig) {
                    edge_indices.push(i_sig);
                }
            }
        }
        edge_indices.sort_unstable();
        edge_indices
    }

    fn build_e_surface_for_edges<T: FloatLike>(
        &self,
        esurf_basis_edge_ids: &Vec<usize>,
        edge_ids: &Vec<usize>,
        cache: &ComputationCache<T>,
        loop_momenta: &Vec<LorentzVector<T>>,
        e_shift: T,
        side: usize,
    ) -> GenericESurfaceCache<T> {
        // Build the E-surface corresponding to this Cutkosky cut
        let mut ps = vec![];
        let mut sigs = vec![];
        let mut ms = vec![];
        let cut_basis_indices = self.loop_indices_in_edge_ids(esurf_basis_edge_ids);

        for e_id in edge_ids.iter() {
            ms.push(Into::<T>::into(
                self.supergraph.edges[*e_id].mass * self.supergraph.edges[*e_id].mass,
            ));
            let sig_vec = cut_basis_indices
                .iter()
                .map(|i_sig| {
                    Into::<T>::into((self.supergraph.edges[*e_id].signature.0[*i_sig]) as f64)
                })
                .collect::<Vec<_>>();
            sigs.push(sig_vec);
            let mut shift = LorentzVector {
                t: T::zero(),
                x: T::zero(),
                y: T::zero(),
                z: T::zero(),
            };
            for (i_loop, s) in self.supergraph.edges[*e_id].signature.0.iter().enumerate() {
                if !cut_basis_indices.contains(&i_loop) {
                    shift += loop_momenta[i_loop] * Into::<T>::into((*s) as f64);
                }
            }
            for (i_ext, s) in self.supergraph.edges[*e_id].signature.1.iter().enumerate() {
                shift += cache.external_momenta[i_ext] * Into::<T>::into((*s) as f64);
            }
            ps.push([shift.x, shift.y, shift.z]);
        }
        GenericESurfaceCache::new_from_inputs(cut_basis_indices, sigs, ps, ms, e_shift, side)
    }

    fn build_e_surfaces<T: FloatLike>(
        &self,
        onshell_edge_momenta: &Vec<LorentzVector<T>>,
        cache: &ComputationCache<T>,
        amplitude: &Amplitude,
        e_surfaces: &Vec<Esurface>,
        loop_momenta: &Vec<LorentzVector<T>>,
        side: usize,
    ) -> Vec<GenericESurfaceCache<T>> {
        let mut e_surf_caches: Vec<GenericESurfaceCache<T>> = vec![];

        // Negative edge ids indicate here momenta external to the whole supergraph
        let mut amplitude_externals = vec![];
        for externals_components in amplitude.external_edge_id_and_flip.iter() {
            let mut new_external = LorentzVector {
                t: T::zero(),
                x: T::zero(),
                y: T::zero(),
                z: T::zero(),
            };
            for (e_id, flip) in externals_components {
                new_external += if *e_id < 0 {
                    cache.external_momenta[(-*e_id - 1) as usize] * Into::<T>::into(*flip as f64)
                } else {
                    onshell_edge_momenta[*e_id as usize] * Into::<T>::into(*flip as f64)
                };
            }
            amplitude_externals.push(new_external);
        }

        for e_surf in e_surfaces.iter() {
            let mut e_shift: T = T::zero();
            for (i_ext, sig) in e_surf.shift.iter().enumerate() {
                e_shift += amplitude_externals[i_ext].t * Into::<T>::into(*sig as f64);
            }
            let mut e_surf_cache = self.build_e_surface_for_edges(
                &amplitude.lmb_edges.iter().map(|e| e.id).collect::<Vec<_>>(),
                &e_surf.edge_ids,
                cache,
                loop_momenta,
                e_shift,
                side,
            );
            (e_surf_cache.exists, e_surf_cache.pinched) = e_surf_cache.does_exist();
            e_surf_cache.eval = e_surf_cache.eval(&loop_momenta);
            if e_surf_cache.exists && false {
                e_surf_cache.t_scaling = e_surf_cache.compute_t_scaling(&loop_momenta);
            }
            e_surf_caches.push(e_surf_cache);
        }

        e_surf_caches
    }

    fn build_cts_for_e_surf_id<T: FloatLike>(
        &mut self,
        side: usize,
        e_surf_id: usize,
        scaled_loop_momenta_in_sampling_basis: &Vec<LorentzVector<T>>,
        e_surf_cache: &[Vec<GenericESurfaceCache<T>>; 2],
        cache: &ComputationCache<T>,
        i_cut: usize,
    ) -> Vec<ESurfaceCT<T, GenericESurfaceCache<T>>> {
        let cut = &self.supergraph.cuts[i_cut];
        // Quite some gynmastic needs to take place in order to dynamically build the right basis for solving this E-surface CT
        // I semi-hard-code it for now
        // In general this is more complicated and would involve an actual change of basis, but here we can do it like this
        let loop_indices_for_this_ct = e_surf_cache[side][e_surf_id].get_e_surface_basis_indices();
        let other_side = if side == LEFT { RIGHT } else { LEFT };
        let amplitudes_pair = [&cut.left_amplitude, &cut.right_amplitude];
        let other_side_loop_indices = self.loop_indices_in_edge_ids(
            &amplitudes_pair[other_side]
                .lmb_edges
                .iter()
                .map(|e| e.id)
                .collect::<Vec<_>>(),
        );
        let mut ct_basis_signature = vec![];
        for loop_index in loop_indices_for_this_ct.iter() {
            let mut basis_element = vec![0; self.supergraph.n_loop];
            basis_element[*loop_index] = 1;
            ct_basis_signature.push(basis_element);
        }

        // Same for the center coordinates, the whole convex solver will need to be implemented for this.
        // And of course E-surf multi-channeling on top if there needs to be multiple centers.
        let c = &self
            .integrand_settings
            .threshold_ct_settings
            .parameterization_center;

        let mut center_coordinates = vec![[T::zero(); 3]; self.supergraph.n_loop];
        for loop_index in loop_indices_for_this_ct.iter() {
            center_coordinates[*loop_index] = [
                Into::<T>::into(c[*loop_index][0]),
                Into::<T>::into(c[*loop_index][1]),
                Into::<T>::into(c[*loop_index][2]),
            ];
        }
        for loop_index in other_side_loop_indices.iter() {
            center_coordinates[*loop_index] = [
                Into::<T>::into(c[*loop_index][0]),
                Into::<T>::into(c[*loop_index][1]),
                Into::<T>::into(c[*loop_index][2]),
            ];
        }

        let mut center_shifts = vec![];
        for center_coordinate in center_coordinates.iter() {
            center_shifts.push(LorentzVector {
                t: T::zero(),
                x: center_coordinate[0],
                y: center_coordinate[1],
                z: center_coordinate[2],
            });
        }

        let mut loop_momenta_in_e_surf_basis = scaled_loop_momenta_in_sampling_basis.clone();
        for loop_index in loop_indices_for_this_ct.iter() {
            loop_momenta_in_e_surf_basis[*loop_index] -= center_shifts[*loop_index];
        }

        // The building of the E-surface should be done more generically and efficiently, but here in this simple case we can do it this way
        let mut subtracted_e_surface = e_surf_cache[side][e_surf_id].clone();

        let center_eval = subtracted_e_surface.eval(&center_shifts);
        assert!(center_eval < T::zero());

        // Change the parametric equation of the subtracted E-surface to the CT basis
        subtracted_e_surface.adjust_loop_momenta_shifts(&center_coordinates);

        subtracted_e_surface.t_scaling =
            subtracted_e_surface.compute_t_scaling(&loop_momenta_in_e_surf_basis);
        if subtracted_e_surface.t_scaling[MINUS] > T::zero() {
            panic!(
                "Unexpected positive t-scaling for negative solution: {:+.e}",
                subtracted_e_surface.t_scaling[MINUS]
            );
        }
        if subtracted_e_surface.t_scaling[PLUS] < T::zero() {
            panic!(
                "Unexpected negative t-scaling for positive solution: {:+.e}",
                subtracted_e_surface.t_scaling[PLUS]
            );
        }
        assert!(subtracted_e_surface.t_scaling[MINUS] <= T::zero());
        assert!(subtracted_e_surface.t_scaling[PLUS] >= T::zero());

        let mut all_new_cts = vec![];

        let local_ct_sliver_width = self
            .integrand_settings
            .threshold_ct_settings
            .local_ct_sliver_width
            .map(|s| Into::<T>::into(s));
        let integrated_ct_sliver_width = self
            .integrand_settings
            .threshold_ct_settings
            .integrated_ct_sliver_width
            .map(|s| Into::<T>::into(s));
        let solutions_to_consider = match self.integrand_settings.threshold_ct_settings.variable {
            CTVariable::Radius => {
                if let Some(s) = local_ct_sliver_width {
                    if s > T::one() {
                        vec![PLUS, MINUS]
                    } else {
                        vec![PLUS]
                    }
                } else {
                    vec![PLUS, MINUS]
                }
            }
            CTVariable::LogRadius => vec![PLUS],
        };

        let ct_levels_to_consider = if self
            .integrand_settings
            .threshold_ct_settings
            .im_squared_through_local_ct_only
        {
            if self
                .integrand_settings
                .threshold_ct_settings
                .include_amplitude_level_cts
            {
                vec![AMPLITUDE_LEVEL_CT, SUPERGRAPH_LEVEL_CT]
            } else {
                vec![SUPERGRAPH_LEVEL_CT]
            }
        } else {
            vec![AMPLITUDE_LEVEL_CT]
        };

        for ct_level in ct_levels_to_consider {
            loop_momenta_in_e_surf_basis = scaled_loop_momenta_in_sampling_basis.clone();
            for loop_index in loop_indices_for_this_ct.iter() {
                loop_momenta_in_e_surf_basis[*loop_index] -= center_shifts[*loop_index];
            }
            if ct_level == SUPERGRAPH_LEVEL_CT {
                for loop_index in other_side_loop_indices.iter() {
                    loop_momenta_in_e_surf_basis[*loop_index] -= center_shifts[*loop_index];
                }
            }
            for solution_type in solutions_to_consider.clone() {
                // println!(
                //     "t considered = {:+.e}=",
                //     subtracted_e_surface.t_scaling[solution_type]
                // );
                let mut loop_momenta_star_in_e_surf_basis = loop_momenta_in_e_surf_basis.clone();
                for loop_index in loop_indices_for_this_ct.iter() {
                    loop_momenta_star_in_e_surf_basis[*loop_index] *=
                        subtracted_e_surface.t_scaling[solution_type];
                }
                if ct_level == SUPERGRAPH_LEVEL_CT {
                    for loop_index in other_side_loop_indices.iter() {
                        loop_momenta_star_in_e_surf_basis[*loop_index] *=
                            subtracted_e_surface.t_scaling[solution_type];
                    }
                }
                // println!(
                //     "loop_momenta_star_in_e_surf_basis={:?}",
                //     loop_momenta_star_in_e_surf_basis
                // );
                let mut loop_momenta_star_in_sampling_basis =
                    loop_momenta_star_in_e_surf_basis.clone();
                for loop_index in loop_indices_for_this_ct.iter() {
                    loop_momenta_star_in_sampling_basis[*loop_index] += center_shifts[*loop_index];
                }
                if ct_level == SUPERGRAPH_LEVEL_CT {
                    for loop_index in other_side_loop_indices.iter() {
                        loop_momenta_star_in_sampling_basis[*loop_index] +=
                            center_shifts[*loop_index];
                    }
                }
                // println!(
                //     "loop_momenta_star_in_sampling_basis={:?}",
                //     loop_momenta_star_in_sampling_basis
                // );

                let mut r_star = loop_indices_for_this_ct
                    .iter()
                    .map(|i| {
                        loop_momenta_star_in_e_surf_basis[*i]
                            .spatial_squared()
                            .abs()
                    })
                    .sum::<T>();
                if ct_level == SUPERGRAPH_LEVEL_CT {
                    r_star += other_side_loop_indices
                        .iter()
                        .map(|i| {
                            loop_momenta_star_in_e_surf_basis[*i]
                                .spatial_squared()
                                .abs()
                        })
                        .sum::<T>();
                }
                r_star = r_star.sqrt();

                let r = r_star / subtracted_e_surface.t_scaling[solution_type];
                // println!("r = {}", r);
                // println!("r_star = {}", r_star);

                let (mut t, mut t_star) =
                    match self.integrand_settings.threshold_ct_settings.variable {
                        CTVariable::Radius => (r, r_star),
                        CTVariable::LogRadius => (r.ln(), r_star.ln()),
                    };

                let mut include_local_ct = true;
                if let Some(sliver) = local_ct_sliver_width {
                    if ((t - t_star) / t_star) * ((t - t_star) / t_star) > sliver * sliver {
                        include_local_ct = false;
                    }
                };
                if ct_level == SUPERGRAPH_LEVEL_CT && !include_local_ct {
                    continue;
                }
                let mut include_integrated_ct = ct_level == AMPLITUDE_LEVEL_CT
                    && solution_type == PLUS
                    && self
                        .integrand_settings
                        .threshold_ct_settings
                        .include_integrated_ct;
                if include_integrated_ct {
                    if let Some(sliver) = integrated_ct_sliver_width {
                        if sliver > T::one() {
                            panic!("{}", format!("{}","It is typically unsafe to set the integrated CT sliver width to be larger \
                        than around one.
                        This is because this region above will typically be undersampled.
                        The result will be inacurate with a bias not represented in the MC accuracy.
                        Comment this check in the code if you really want to proceed.").red());
                        }
                        if ((t - t_star) / t_star) * ((t - t_star) / t_star) > sliver * sliver {
                            include_integrated_ct = false;
                        }
                    }
                }
                if !include_local_ct && !include_integrated_ct {
                    continue;
                }

                let onshell_edge_momenta_for_this_ct = self.evaluate_onshell_edge_momenta(
                    &loop_momenta_star_in_sampling_basis,
                    &cache.external_momenta,
                    cut,
                );

                if self
                    .integrand_settings
                    .threshold_ct_settings
                    .apply_anti_observable
                {
                    //TODO: the implementation below is of course dummy and should be replaced by the actual anti-observable
                    //for now we'll call it unimplemented
                    unimplemented!("The anti-observable is not implemented yet");
                    let ct_cut_edges_and_flip = vec![(1_usize, -1_isize)];
                    let mut evt = self.event_manager.create_event(
                        vec![cache.external_momenta[0]],
                        ct_cut_edges_and_flip
                            .iter()
                            .map(|(i_edge, flip)| {
                                onshell_edge_momenta_for_this_ct[*i_edge]
                                    * Into::<T>::into(*flip as f64)
                            })
                            .collect::<Vec<_>>(),
                    );
                    if !self.event_manager.pass_selection(&mut evt) {
                        continue;
                    }
                }
                // Update the evaluation of the E-surface for the solved star loop momentum in the sampling basis (since it is what the shifts are computed for in this cache)
                let mut e_surface_caches_for_this_ct = e_surf_cache.clone();
                for e_surf in e_surface_caches_for_this_ct[side].iter_mut() {
                    e_surf.eval = e_surf.eval(&loop_momenta_star_in_sampling_basis);
                }
                if ct_level == SUPERGRAPH_LEVEL_CT {
                    for e_surf in e_surface_caches_for_this_ct[other_side].iter_mut() {
                        e_surf.eval = e_surf.eval(&loop_momenta_star_in_sampling_basis);
                    }
                }

                // The factor (t - t_star) will be included at the end because we need the same quantity for the integrated CT
                // let e_surf_derivative = e_surf_cache[side][e_surf_id]
                //     .norm(&vec![
                //         loop_momenta_star_in_sampling_basis[loop_index_for_this_ct],
                //     ])
                //     .spatial_dot(&loop_momenta_star_in_e_surf_basis[loop_index_for_this_ct])
                let e_surf_derivative = e_surf_cache[side][e_surf_id]
                    .t_der(&loop_momenta_star_in_sampling_basis)
                    / r_star;

                // Identifying the residue in t, with r=e^t means that we must drop the r_star normalisation in the expansion.
                let e_surf_expanded = match self.integrand_settings.threshold_ct_settings.variable {
                    CTVariable::Radius => e_surf_derivative * (t - t_star),
                    CTVariable::LogRadius => e_surf_derivative * r_star * (t - t_star),
                };

                let h_function_wgt = utils::h(
                    t,
                    Some(t_star),
                    None,
                    &self
                        .integrand_settings
                        .threshold_ct_settings
                        .local_ct_h_function,
                );
                // Start with r / r_star in order to implement the power offset of -1.
                let mut adjusted_sampling_jac = (r / r_star);
                // Now account for the radius impact on the jacobian.
                for _ in 0..loop_indices_for_this_ct.len() {
                    adjusted_sampling_jac *= (r_star / r).powi(3);
                }
                if ct_level == SUPERGRAPH_LEVEL_CT {
                    for _ in 0..other_side_loop_indices.len() {
                        adjusted_sampling_jac *= (r_star / r).powi(3);
                    }
                }
                // Disable the local CT by setting the weight of the adjusted_sampling_jac to zero.
                // We cannot skip any computation here because they are needed for the computation of the integrated CT which is
                // not disabled if this point in the code is reached.
                if !include_local_ct {
                    adjusted_sampling_jac = T::zero();
                }
                // The measure when building the local CT for the variable t, with r=e^t, means that we must adjust for one more power,
                // because dr = r dt.
                match self.integrand_settings.threshold_ct_settings.variable {
                    CTVariable::Radius => {}
                    CTVariable::LogRadius => {
                        adjusted_sampling_jac *= r_star / r;
                    }
                }

                let mut cff_evaluations = [vec![], vec![]];
                let mut cff_pinch_dampenings = [vec![], vec![]];
                let dampening_normalisation_factor = Into::<T>::into(self.settings.kinematics.e_cm);
                // let (dampening_t_arg, dampening_t_star_arg) =
                //     match self.integrand_settings.threshold_ct_settings.variable {
                //         CTVariable::Radius => (
                //             t / dampening_normalisation_factor,
                //             t_star / dampening_normalisation_factor,
                //         ),
                //         CTVariable::LogRadius => (t, t_star),
                //     };
                let dampening_delta_t_arg =
                    match self.integrand_settings.threshold_ct_settings.variable {
                        CTVariable::Radius => (t - t_star) / t_star,
                        CTVariable::LogRadius => (t - t_star) / t_star,
                    };
                let pinch_dampening_settings = &self
                    .integrand_settings
                    .threshold_ct_settings
                    .pinch_dampening;
                let mut global_pinch_dampening_ct_side = T::one();
                let mut global_pinch_dampening_other_side = T::one();
                if pinch_dampening_settings.enabled && pinch_dampening_settings.global {
                    // Also add pinched E-surface from non-loop propagators if dampen_all_pinch_surfaces if true
                    // This is a bit ad-hoc and should be done more in general for the actual code
                    // Of course we also take for granted that everything is massless here.
                    for prop_signature in &amplitudes_pair[side].non_loop_propagators {
                        let prop_momentum = compute_propagator_momentum(
                            prop_signature,
                            &onshell_edge_momenta_for_this_ct,
                            &cache,
                        );
                        let prop_e_surf_eval =
                            prop_momentum.t.abs() - prop_momentum.spatial_distance();
                        global_pinch_dampening_ct_side *= utils::pinch_dampening_function(
                            prop_e_surf_eval / dampening_normalisation_factor,
                            dampening_delta_t_arg,
                            pinch_dampening_settings.powers,
                            pinch_dampening_settings.regularization_multiplier,
                        );
                    }
                    for e_surf in e_surf_cache[side].iter() {
                        if e_surf.pinched {
                            if !pinch_dampening_settings.dampen_all_pinch_surfaces {
                                if !subtracted_e_surface.overlaps_with(e_surf) {
                                    continue;
                                }
                            }
                            global_pinch_dampening_ct_side *= utils::pinch_dampening_function(
                                e_surf.eval / dampening_normalisation_factor,
                                dampening_delta_t_arg,
                                pinch_dampening_settings.powers,
                                pinch_dampening_settings.regularization_multiplier,
                            )
                        }
                    }
                    if pinch_dampening_settings.dampen_all_pinch_surfaces
                        && ct_level == SUPERGRAPH_LEVEL_CT
                    {
                        for prop_signature in &amplitudes_pair[other_side].non_loop_propagators {
                            let prop_momentum = compute_propagator_momentum(
                                prop_signature,
                                &onshell_edge_momenta_for_this_ct,
                                &cache,
                            );
                            let prop_e_surf_eval =
                                prop_momentum.t.abs() - prop_momentum.spatial_distance();
                            global_pinch_dampening_other_side *= utils::pinch_dampening_function(
                                prop_e_surf_eval / dampening_normalisation_factor,
                                dampening_delta_t_arg,
                                pinch_dampening_settings.powers,
                                pinch_dampening_settings.regularization_multiplier,
                            );
                        }
                        for e_surf in e_surf_cache[other_side].iter() {
                            if e_surf.pinched {
                                global_pinch_dampening_other_side *= utils::pinch_dampening_function(
                                    e_surf.eval / dampening_normalisation_factor,
                                    dampening_delta_t_arg,
                                    pinch_dampening_settings.powers,
                                    pinch_dampening_settings.regularization_multiplier,
                                )
                            }
                        }
                    }
                }
                // println!(
                //     "global_pinch_dampening_ct_side = {}",
                //     global_pinch_dampening_ct_side
                // );
                // println!(
                //     "global_pinch_dampening_other_side = {}",
                //     global_pinch_dampening_other_side
                // );
                for (i_cff, cff_term) in amplitudes_pair[side]
                    .cff_expression
                    .terms
                    .iter()
                    .enumerate()
                {
                    if !cff_term.contains_e_surf_id(e_surf_id) {
                        cff_evaluations[side].push(T::zero());
                        cff_pinch_dampenings[side].push(T::one());
                    } else {
                        cff_evaluations[side].push(cff_term.evaluate(
                            &e_surface_caches_for_this_ct[side],
                            Some(vec![(e_surf_id, T::one())]),
                        ));
                        if !pinch_dampening_settings.enabled || pinch_dampening_settings.global {
                            if ct_level == AMPLITUDE_LEVEL_CT {
                                cff_pinch_dampenings[side].push(
                                    global_pinch_dampening_ct_side
                                        * global_pinch_dampening_other_side,
                                );
                            } else {
                                cff_pinch_dampenings[side].push(global_pinch_dampening_ct_side);
                            }
                        } else {
                            let mut dampening = T::one();
                            for (i_surf, e_surf) in e_surf_cache[side].iter().enumerate() {
                                if e_surf.pinched && cff_term.contains_e_surf_id(i_surf) {
                                    if !pinch_dampening_settings.dampen_all_pinch_surfaces {
                                        if !subtracted_e_surface.overlaps_with(e_surf) {
                                            continue;
                                        }
                                    }
                                    dampening *= utils::pinch_dampening_function(
                                        e_surf.eval / dampening_normalisation_factor,
                                        dampening_delta_t_arg,
                                        pinch_dampening_settings.powers,
                                        pinch_dampening_settings.regularization_multiplier,
                                    )
                                }
                            }
                            cff_pinch_dampenings[side].push(dampening);
                        }
                    }
                }
                if ct_level == SUPERGRAPH_LEVEL_CT {
                    for (i_cff, cff_term) in amplitudes_pair[other_side]
                        .cff_expression
                        .terms
                        .iter()
                        .enumerate()
                    {
                        cff_evaluations[other_side].push(
                            cff_term.evaluate(&e_surface_caches_for_this_ct[other_side], None),
                        );
                        if !pinch_dampening_settings.enabled || pinch_dampening_settings.global {
                            cff_pinch_dampenings[other_side]
                                .push(global_pinch_dampening_other_side);
                        } else {
                            if pinch_dampening_settings.dampen_all_pinch_surfaces {
                                let mut dampening = T::one();
                                for (i_surf, e_surf) in e_surf_cache[other_side].iter().enumerate()
                                {
                                    if e_surf.pinched && cff_term.contains_e_surf_id(i_surf) {
                                        dampening *= utils::pinch_dampening_function(
                                            e_surf.eval / dampening_normalisation_factor,
                                            dampening_delta_t_arg,
                                            pinch_dampening_settings.powers,
                                            pinch_dampening_settings.regularization_multiplier,
                                        )
                                    }
                                }
                                cff_pinch_dampenings[other_side].push(dampening);
                            } else {
                                cff_pinch_dampenings[other_side].push(T::one());
                            }
                        }
                    }
                }

                let integrated_ct = if include_integrated_ct {
                    // Keep in mind that the suface element d S_r keeps an r^2 fact, but that of the solved radius, so we
                    // still need this correction factor.
                    // So it happens to be the same correction factor as for the local CT but it's not immediatlly obvious so I keep them separate
                    // Start with r / r_star in order to implement the power offset of -1.
                    let mut ict_adjusted_sampling_jac = (r / r_star);
                    // Now account for the radius impact on the jacobian.
                    for _ in 0..loop_indices_for_this_ct.len() {
                        ict_adjusted_sampling_jac *= (r_star / r).powi(3);
                    }
                    // Notice that for the arguments of the h function of the iCT, once can put pretty much anything that goes from 0 to infty.
                    // It would all work, but it's likely best to have an argument of one when the integrator samples directly the e-surface.
                    let ict_h_function = if let Some(sliver) = integrated_ct_sliver_width {
                        match self.integrand_settings.threshold_ct_settings.variable {
                            CTVariable::Radius => {
                                (Into::<T>::into(2 as f64) * sliver * r_star).inv()
                            }
                            CTVariable::LogRadius => (r_star.powf(T::one() + sliver)
                                - r_star.powf(T::one() - sliver))
                            .inv(),
                        }
                    } else {
                        r_star.inv()
                            * utils::h(
                                r / r_star,
                                None,
                                None, // Some(radius_star_in_sampling_basis*radius_star_in_sampling_basis)
                                &self
                                    .integrand_settings
                                    .threshold_ct_settings
                                    .integrated_ct_h_function,
                            )
                    };

                    // TODO investigate factor 1/2, why is it needed?
                    let e_surf_residue = e_surf_derivative.inv()
                        * (Into::<T>::into(2 as f64) * <T as FloatConst>::PI())
                        / Into::<T>::into(2 as f64);
                    Some(ESurfaceIntegratedCT {
                        adjusted_sampling_jac: ict_adjusted_sampling_jac,
                        h_function_wgt: ict_h_function,
                        e_surf_residue: e_surf_residue,
                    })
                } else {
                    None
                };

                all_new_cts.push(ESurfaceCT {
                    e_surf_id,
                    ct_basis_signature: ct_basis_signature.clone(), // Not used at the moment, could be dropped
                    center_coordinates: center_coordinates.clone(), // Not used at the moment, could be dropped
                    adjusted_sampling_jac,
                    h_function_wgt,
                    e_surf_expanded: e_surf_expanded,
                    loop_momenta_star: loop_momenta_star_in_sampling_basis, // Not used at the moment, could be dropped
                    onshell_edges: onshell_edge_momenta_for_this_ct,
                    e_surface_evals: e_surface_caches_for_this_ct, // Not used at the moment, could be dropped
                    solution_type,                                 // Only for monitoring purposes
                    cff_evaluations,
                    cff_pinch_dampenings,
                    integrated_ct,
                    ct_level,
                });
            }
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
        &mut self,
        i_cut: usize,
        loop_momenta: &Vec<LorentzVector<T>>,
        overall_sampling_jac: T,
        cache: &ComputationCache<T>,
    ) -> (
        Complex<T>,
        Complex<T>,
        Option<ClosestESurfaceMonitor<T>>,
        Option<ClosestESurfaceMonitor<T>>,
    ) {
        //let cut = &self.supergraph.cuts[i_cut];
        let mut cut_res = Complex::new(T::zero(), T::zero());
        if self.settings.general.debug > 1 {
            println!(
                "{}",
                format!(
                "  > Starting evaluation of cut #{}{} ( n_loop_left={} | cut_cardinality={} | n_loop_right={} )",
                i_cut,
                format!("({})", self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 { format!("+{}",id) } else { format!("-{}",id) }).collect::<Vec<_>>().join("|")).blue(),
                self.supergraph.cuts[i_cut].left_amplitude.n_loop,
                self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.len(),
                self.supergraph.cuts[i_cut].right_amplitude.n_loop,            )
                .blue()
            );
        }

        // Include constants and flux factor
        let mut constants = Complex::new(T::one(), T::zero());
        constants /= Into::<T>::into(2 as f64) * cache.external_momenta[0].square().sqrt().abs();
        // And the 2 pi for each edge
        constants /= (Into::<T>::into(2 as f64) * T::PI()).powi(self.supergraph.edges.len() as i32);

        // Evaluate kinematics before forcing correct hyperradius
        let onshell_edge_momenta_for_this_cut = self.evaluate_onshell_edge_momenta(
            &loop_momenta,
            &cache.external_momenta,
            &self.supergraph.cuts[i_cut],
        );

        // Build the E-surface corresponding to this Cutkosky cut
        let cut_edge_ids = &self.supergraph.cuts[i_cut]
            .cut_edge_ids_and_flip
            .iter()
            .map(|(e_id, _flip)| *e_id)
            .collect::<Vec<_>>();
        let mut e_surface_cc_cut = self.build_e_surface_for_edges(
            cut_edge_ids,
            cut_edge_ids,
            &cache,
            // Nothing will be used from the loop momenta in this context because we specify all edges to be in the e surf basis here.
            // But this construction is useful when building the amplitude e-surfaces using the same function.
            loop_momenta,
            -cache.external_momenta[0].t,
            LEFT,
        );

        e_surface_cc_cut.t_scaling = e_surface_cc_cut.compute_t_scaling(&loop_momenta);

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
                "    Rescaling for this cut: {:+.e}",
                e_surface_cc_cut.t_scaling[0]
            );
            println!(
                "    Normalised hyperradius for this cut: {:+.e}",
                normalised_hyperradius
            );
            println!(
                "    t-scaling jacobian and h-function for this cut: {:+.e}",
                t_scaling_jacobian * cut_h_function
            );
        }

        // Include the t-derivative of the E-surface as well
        let cut_e_surface_derivative =
            e_surface_cc_cut.t_scaling[0] / e_surface_cc_cut.t_der(&rescaled_loop_momenta);

        // Now re-evaluate the kinematics with the correct hyperradius
        let onshell_edge_momenta_for_this_cut = self.evaluate_onshell_edge_momenta(
            &rescaled_loop_momenta,
            &cache.external_momenta,
            &self.supergraph.cuts[i_cut],
        );

        // Now add the event and check if it passes the selector
        let mut evt = self.event_manager.create_event(
            vec![cache.external_momenta[0]],
            self.supergraph.cuts[i_cut]
                .cut_edge_ids_and_flip
                .iter()
                .map(|(i_edge, flip)| {
                    onshell_edge_momenta_for_this_cut[*i_edge] * Into::<T>::into(*flip as f64)
                })
                .collect::<Vec<_>>(),
        );
        if !self.event_manager.add_event(evt) {
            if self.settings.general.debug > 0 {
                println!(
                    "The following event for cut #{}{} failed to pass the cuts:\n{:?}",
                    i_cut,
                    format!(
                        "({})",
                        self.supergraph.cuts[i_cut]
                            .cut_edge_ids_and_flip
                            .iter()
                            .map(|(id, flip)| if *flip > 0 {
                                format!("+{}", id)
                            } else {
                                format!("-{}", id)
                            })
                            .collect::<Vec<_>>()
                            .join("|")
                    ),
                    self.event_manager.create_event(
                        cache.external_momenta.clone(),
                        self.supergraph.cuts[i_cut]
                            .cut_edge_ids_and_flip
                            .iter()
                            .map(|(i_edge, flip)| {
                                onshell_edge_momenta_for_this_cut[*i_edge]
                                    * Into::<T>::into(*flip as f64)
                            })
                            .collect::<Vec<_>>(),
                    )
                );
            }
            return (
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::zero(), T::zero()),
                None,
                None,
            );
        }

        if self.settings.general.debug > 2 {
            println!("    Edge on-shell momenta for this cut:");
            for (i, l) in onshell_edge_momenta_for_this_cut.iter().enumerate() {
                println!(
                    "      {} = ( {:-45}, {:-45}, {:-45}, {:-45} )", // sqrt(q{}^2)={:+.e}",
                    format!("q{}", i).bold().green(),
                    format!("{:+.e}", l.t),
                    format!("{:+.e}", l.x),
                    format!("{:+.e}", l.y),
                    format!("{:+.e}", l.z) // i, l.square().abs().sqrt()
                );
            }
        }

        // Evaluate E-surfaces
        let mut e_surf_caches: [Vec<GenericESurfaceCache<T>>; 2] = [vec![], vec![]];
        for side in [LEFT, RIGHT] {
            let amplitude_for_sides = [
                &self.supergraph.cuts[i_cut].left_amplitude,
                &self.supergraph.cuts[i_cut].right_amplitude,
            ];
            e_surf_caches[side] = self.build_e_surfaces(
                &onshell_edge_momenta_for_this_cut,
                &cache,
                &amplitude_for_sides[side],
                &amplitude_for_sides[side].cff_expression.e_surfaces,
                &rescaled_loop_momenta,
                side,
            );
        }
        let mut closest_existing_e_surf: Option<ClosestESurfaceMonitor<T>> = None;
        let mut closest_pinched_e_surf: Option<ClosestESurfaceMonitor<T>> = None;
        if self.settings.general.debug > 0 {
            let amplitude_for_sides = [
                &self.supergraph.cuts[i_cut].left_amplitude,
                &self.supergraph.cuts[i_cut].right_amplitude,
            ];
            for side in [LEFT, RIGHT] {
                for (i_surf, e_surf_cache) in e_surf_caches[side].iter().enumerate() {
                    let new_monitor = ClosestESurfaceMonitor {
                        distance: e_surf_cache.eval
                            / Into::<T>::into(self.settings.kinematics.e_cm as f64),
                        e_surf_id: i_surf,
                        i_cut: i_cut,
                        side: side,
                        e_surf: amplitude_for_sides[side].cff_expression.e_surfaces[i_surf].clone(),
                        e_surf_cache: e_surf_cache.clone(),
                    };
                    if e_surf_cache.exists && !e_surf_cache.pinched {
                        if let Some(closest) = &closest_existing_e_surf {
                            if new_monitor.distance.abs() < closest.distance.abs() {
                                closest_existing_e_surf = Some(new_monitor);
                            }
                        } else {
                            closest_existing_e_surf = Some(new_monitor)
                        }
                    } else if e_surf_cache.pinched {
                        if let Some(closest) = &closest_pinched_e_surf {
                            if new_monitor.distance.abs() < closest.distance.abs() {
                                closest_pinched_e_surf = Some(new_monitor);
                            }
                        } else {
                            closest_pinched_e_surf = Some(new_monitor)
                        }
                    }
                }
            }
        }
        if self.settings.general.debug > 3 {
            let amplitude_for_sides = [
                &self.supergraph.cuts[i_cut].left_amplitude,
                &self.supergraph.cuts[i_cut].right_amplitude,
            ];
            for side in [LEFT, RIGHT] {
                if e_surf_caches[side].len() == 0 {
                    println!(
                        "    All {} e_surf caches side:      {}",
                        format!("{}", if side == LEFT { "left" } else { "right" }).purple(),
                        format!("{}", "None").green()
                    );
                } else {
                    let mut n_e_surf_to_subtract = 0;
                    for es in e_surf_caches[side].iter() {
                        if es.exists && !es.pinched {
                            n_e_surf_to_subtract += 1;
                        }
                    }
                    println!(
                        "    All {} e_surf caches side {}:\n      {}",
                        format!("{}", if side == LEFT { "left" } else { "right" }).purple(),
                        if n_e_surf_to_subtract > 0 {
                            format!(
                                "({})",
                                format!(
                                    "a total of {} E-surfaces requiring counterterms are displayed in red",
                                    n_e_surf_to_subtract
                                )
                                .bold()
                                .red(),
                            )
                        } else {
                            format!(
                                "({})",
                                format!("{}", "no E-surface requires counterterms.")
                                    .bold()
                                    .green()
                            )
                        },
                        e_surf_caches[side]
                            .iter()
                            .enumerate()
                            .map(|(i_esurf, es)| format!(
                                "{} : {:?}",
                                if es.exists && !es.pinched {
                                    e_surf_str(
                                        i_esurf,
                                        &amplitude_for_sides[side].cff_expression.e_surfaces
                                            [i_esurf],
                                    )
                                    .bold()
                                    .red()
                                } else {
                                    e_surf_str(
                                        i_esurf,
                                        &amplitude_for_sides[side].cff_expression.e_surfaces
                                            [i_esurf],
                                    )
                                    .bold()
                                    .green()
                                },
                                es
                            ))
                            .collect::<Vec<_>>()
                            .join("\n      ")
                    );
                }
            }
        }

        let mut i_term = 0;
        let mut cff_evaluations = [vec![], vec![]];
        for side in [LEFT, RIGHT] {
            let amplitude_for_sides = [
                &self.supergraph.cuts[i_cut].left_amplitude,
                &self.supergraph.cuts[i_cut].right_amplitude,
            ];
            for (i_cff, cff_term) in amplitude_for_sides[side]
                .cff_expression
                .terms
                .iter()
                .enumerate()
            {
                let cff_eval = cff_term.evaluate(&e_surf_caches[side], None);
                if self.settings.general.debug > 2 {
                    println!(
                        "   > {} cFF evaluation for orientation #{:-3}({}): {}",
                        if side == LEFT { "Left " } else { "Right" },
                        format!("{}", i_cff).green(),
                        cff_term
                            .orientation
                            .iter()
                            .map(|(_id, flip)| if *flip > 0 { "+" } else { "-" })
                            .collect::<Vec<_>>()
                            .join("")
                            .blue(),
                        format!("{:+.e}", cff_eval).blue()
                    );
                }
                cff_evaluations[side].push(cff_eval);
            }
        }

        let mut e_product_left = T::one();
        for e in self.supergraph.cuts[i_cut].left_amplitude.edges.iter() {
            e_product_left *=
                Into::<T>::into(2 as f64) * onshell_edge_momenta_for_this_cut[e.id].t.abs();
        }
        let mut e_product_right = T::one();
        for e in self.supergraph.cuts[i_cut].right_amplitude.edges.iter() {
            e_product_right *=
                Into::<T>::into(2 as f64) * onshell_edge_momenta_for_this_cut[e.id].t.abs();
        }
        // Build the non-loop propagators too
        let mut non_loop_propagators_contributions = [T::one(), T::one()];
        for side in [LEFT, RIGHT] {
            let amplitude_for_sides = [
                &self.supergraph.cuts[i_cut].left_amplitude,
                &self.supergraph.cuts[i_cut].right_amplitude,
            ];
            for non_loop_prop_signature in amplitude_for_sides[side].non_loop_propagators.iter() {
                let prop_momentum = compute_propagator_momentum(
                    non_loop_prop_signature,
                    &onshell_edge_momenta_for_this_cut,
                    &cache,
                );
                non_loop_propagators_contributions[side] *= prop_momentum.square();
            }
        }
        non_loop_propagators_contributions[0] = non_loop_propagators_contributions[0].inv();
        non_loop_propagators_contributions[1] = non_loop_propagators_contributions[1].inv();
        let non_loop_propagators_contribution =
            non_loop_propagators_contributions[0] * non_loop_propagators_contributions[1];

        // Now build the counterterms
        let mut cts = [vec![], vec![]];
        if self.integrand_settings.threshold_ct_settings.enabled {
            // There are smarter ways to do this, but this is the most straightforward and clear for this exploration
            for side in [LEFT, RIGHT] {
                let n_e_surfs = if side == LEFT {
                    self.supergraph.cuts[i_cut]
                        .left_amplitude
                        .cff_expression
                        .e_surfaces
                        .len()
                } else {
                    self.supergraph.cuts[i_cut]
                        .right_amplitude
                        .cff_expression
                        .e_surfaces
                        .len()
                };
                for e_surf_id in 0..n_e_surfs {
                    if e_surf_caches[side][e_surf_id].exists
                        && !e_surf_caches[side][e_surf_id].pinched
                    {
                        if self.settings.general.debug > 4 {
                            let amplitude_for_sides = [
                                &self.supergraph.cuts[i_cut].left_amplitude,
                                &self.supergraph.cuts[i_cut].right_amplitude,
                            ];
                            println!(
                                "Now building CTs for the following E-surface {}",
                                format!(
                                    "{} : {:?}",
                                    e_surf_str(
                                        e_surf_id,
                                        &amplitude_for_sides[side].cff_expression.e_surfaces
                                            [e_surf_id]
                                    )
                                    .bold()
                                    .green(),
                                    e_surf_caches[side][e_surf_id]
                                )
                            );
                        }
                        let new_cts = self.build_cts_for_e_surf_id(
                            side,
                            e_surf_id,
                            &rescaled_loop_momenta,
                            &e_surf_caches,
                            cache,
                            i_cut,
                        );
                        cts[side].extend(new_cts);
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

        // We can obtain the imaginary squared contribution using local CTs only as follows:

        // This is the result when using supergraph-level CTs only:
        // A) I - SG_CT = RealPart[ (LAMP.re + i RAMP.im) * (RAMP.re + i RAMP.im) ] = LAMP.re * RAMP.re - RAMP.im * RAMP.im
        // So no complex conjugation sadly.

        // However when using amplitude level local CTs, we have:
        // B) I - AMP_CT = RealPart[ LAMP.re + i RAMP.im ] * RealPart[ RAMP.re + i RAMP.im ] = LAMP.re * LAMP.re

        // And the real part of the quantity we want, with complex conjugation, is:
        // LAMP.re * RAMP.re + RAMP.im * RAMP.im
        // We can therefore obtain the above using: B + (B - A) = 2 B - A
        // Which is what we will be implementing below, i.e.:
        // I - 2 AMP_CT + SG_CT
        let use_imaginary_squared_trick = self
            .integrand_settings
            .threshold_ct_settings
            .im_squared_through_local_ct_only
            && self
                .integrand_settings
                .threshold_ct_settings
                .include_amplitude_level_cts;

        // Note that we could also consider splitting the numerator into a left and right component, depending on its implementation
        let mut cff_sum = Complex::new(T::zero(), T::zero());
        let mut cff_cts_sum = Complex::new(T::zero(), T::zero());
        let mut cff_im_squared_cts_sum = T::zero();
        for (left_i_cff, left_cff_term) in self.supergraph.cuts[i_cut]
            .left_amplitude
            .cff_expression
            .terms
            .iter()
            .enumerate()
        {
            for (right_i_cff, right_cff_term) in self.supergraph.cuts[i_cut]
                .right_amplitude
                .cff_expression
                .terms
                .iter()
                .enumerate()
            {
                i_term += 1;
                let mut numerator_wgt = self.evaluate_numerator(
                    &onshell_edge_momenta_for_this_cut,
                    &left_cff_term.orientation,
                    &right_cff_term.orientation,
                );

                let cff_left_wgt = cff_evaluations[LEFT][left_i_cff];
                let cff_right_wgt = cff_evaluations[RIGHT][right_i_cff];

                let mut this_cff_term_contribution = numerator_wgt
                    * cff_left_wgt
                    * e_product_left.inv()
                    * cff_right_wgt
                    * e_product_right.inv();

                if use_imaginary_squared_trick
                    && self
                        .integrand_settings
                        .threshold_ct_settings
                        .compute_only_im_squared
                {
                    this_cff_term_contribution = T::zero();
                }
                cff_sum += this_cff_term_contribution;

                // Now include counterterms
                let amplitudes_pair = [
                    &self.supergraph.cuts[i_cut].left_amplitude,
                    &self.supergraph.cuts[i_cut].right_amplitude,
                ];
                let i_cff_pair = [left_i_cff, right_i_cff];
                let mut cts_sum_for_this_term = Complex::new(T::zero(), T::zero());
                for ct_side in [LEFT, RIGHT] {
                    let other_side = if ct_side == LEFT { RIGHT } else { LEFT };
                    for ct in cts[ct_side].iter() {
                        if ct.cff_evaluations[ct_side][i_cff_pair[ct_side]] == T::zero() {
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

                        let other_side_terms = if ct.ct_level == AMPLITUDE_LEVEL_CT {
                            if ct_side == LEFT {
                                cff_right_wgt * e_product_right.inv()
                            } else {
                                cff_left_wgt * e_product_left.inv()
                            }
                        } else {
                            let mut e_product = T::one();
                            for e in amplitudes_pair[other_side].edges.iter() {
                                e_product *=
                                    Into::<T>::into(2 as f64) * ct.onshell_edges[e.id].t.abs();
                            }
                            ct.cff_evaluations[other_side][i_cff_pair[other_side]]
                                * ct.cff_pinch_dampenings[other_side][i_cff_pair[other_side]]
                                * e_product.inv()
                        };

                        let mut re_ct_weight = -other_side_terms
                            * ct_numerator_wgt
                            * ct_e_product.inv()
                            * ct.cff_evaluations[ct_side][i_cff_pair[ct_side]]
                            * ct.cff_pinch_dampenings[ct_side][i_cff_pair[ct_side]]
                            * ct.e_surf_expanded.inv()
                            * ct.adjusted_sampling_jac
                            * ct.h_function_wgt;
                        // println!("other_side_terms = {:+.e}", other_side_terms);
                        // println!(
                        //     "ct.adjusted_sampling_jac = {:+.e}",
                        //     ct.adjusted_sampling_jac
                        // );
                        // println!("ct_numerator_wgt = {:+.e}", ct_numerator_wgt);
                        // println!(
                        //     "ct.cff_evaluations[ct_side][i_cff_pair[ct_side]] = {:+.e}",
                        //     ct.cff_evaluations[ct_side][i_cff_pair[ct_side]]
                        // );
                        // println!(
                        //     "ct.cff_pinch_dampenings[ct_side][i_cff_pair[ct_side]] = {:+.e}",
                        //     ct.cff_pinch_dampenings[ct_side][i_cff_pair[ct_side]]
                        // );
                        // println!(
                        //     "ct.cff_evaluations[ct_side][i_cff_pair[ct_side]] = {:+.e}",
                        //     ct.cff_evaluations[ct_side][i_cff_pair[ct_side]]
                        // );
                        // println!(
                        //     "ct.cff_pinch_dampenings[ct_side][i_cff_pair[ct_side]] = {:+.e}",
                        //     ct.cff_pinch_dampenings[ct_side][i_cff_pair[ct_side]]
                        // );
                        // println!("ct.cff_evaluations[*][*] = {:?}", ct.cff_evaluations);
                        // println!("ct type = {}", ct.solution_type);
                        // println!("ct_e_product = {:+.e}", ct_e_product);
                        // println!("e_product_left = {:+.e}", e_product_left);
                        // println!("e_product_right = {:+.e}", e_product_right);
                        // println!("ct.h_function_wgt = {:+.e}", ct.h_function_wgt);
                        // println!("ct.e_surf_derivative_wgt = {:+.e}", ct.e_surf_expanded);
                        // println!(
                        //     "A = {} vs B = {}, A/B = {}",
                        //     cff_left_wgt,
                        //     ct.cff_evaluations[ct_side][i_cff_pair[ct_side]],
                        //     cff_left_wgt / ct.cff_evaluations[ct_side][i_cff_pair[ct_side]],
                        // );
                        let mut im_ct_weight = if let Some(i_ct) = &ct.integrated_ct {
                            other_side_terms
                                * ct_numerator_wgt
                                * ct_e_product.inv()
                                * ct.cff_evaluations[ct_side][i_cff_pair[ct_side]]
                                * i_ct.e_surf_residue
                                * i_ct.adjusted_sampling_jac
                                * i_ct.h_function_wgt
                        } else {
                            T::zero()
                        };

                        // Implement complex-conjugation here
                        if ct_side == RIGHT {
                            im_ct_weight *= -T::one();
                        }

                        // Now depending on user specification, get the imaginary squared part using: I - 2 AMP_CT + SG_CT
                        if use_imaginary_squared_trick {
                            if self
                                .integrand_settings
                                .threshold_ct_settings
                                .compute_only_im_squared
                            {
                                // In that case we only try to integrate B-A which yields the squared imaginary part
                                if ct.ct_level == SUPERGRAPH_LEVEL_CT {
                                    re_ct_weight *= -T::one();
                                }
                            } else {
                                if ct.ct_level == SUPERGRAPH_LEVEL_CT {
                                    re_ct_weight *= -T::one();
                                } else {
                                    re_ct_weight *= Into::<T>::into(2 as f64);
                                }
                            }
                        }

                        if self.settings.general.debug > 3 {
                            let amplitude_for_sides = [
                                &self.supergraph.cuts[i_cut].left_amplitude,
                                &self.supergraph.cuts[i_cut].right_amplitude,
                            ];
                            println!(
                                "   > cFF Evaluation #{} : CT for {} E-surface {} : {:+.e} + i {:+.e}",
                                format!("{}", i_term).green(),
                                format!("{}|{}|{}",
                                if ct_side == LEFT {"L"} else {"R"},
                                if ct.ct_level == AMPLITUDE_LEVEL_CT {"AMP"} else {"SG "},
                                if ct.solution_type == PLUS {"+"} else {"-"}
                                ).purple(),
                                e_surf_str(
                                    ct.e_surf_id,
                                    &amplitude_for_sides[ct_side].cff_expression.e_surfaces[ct.e_surf_id]
                                ).blue(),
                                re_ct_weight,
                                im_ct_weight
                            );
                        }

                        cts_sum_for_this_term += Complex::new(re_ct_weight, im_ct_weight);
                    }
                }

                // Now implement the cross terms
                let mut ct_im_squared_weight_for_this_term = T::zero();
                for left_ct in cts[LEFT].iter() {
                    if left_ct.ct_level == SUPERGRAPH_LEVEL_CT
                        || left_ct.cff_evaluations[LEFT][i_cff_pair[LEFT]] == T::zero()
                    {
                        continue;
                    }
                    for right_ct in cts[RIGHT].iter() {
                        if right_ct.ct_level == SUPERGRAPH_LEVEL_CT
                            || right_ct.cff_evaluations[RIGHT][i_cff_pair[RIGHT]] == T::zero()
                        {
                            continue;
                        }
                        let mut combined_onshell_edges = left_ct.onshell_edges.clone();
                        for edge in &amplitudes_pair[RIGHT].edges {
                            combined_onshell_edges[edge.id] = right_ct.onshell_edges[edge.id];
                        }
                        let ct_numerator_wgt = self.evaluate_numerator(
                            &combined_onshell_edges,
                            &left_cff_term.orientation,
                            &right_cff_term.orientation,
                        );
                        let mut ct_e_product = T::one();
                        for e in amplitudes_pair[LEFT].edges.iter() {
                            ct_e_product *=
                                Into::<T>::into(2 as f64) * left_ct.onshell_edges[e.id].t.abs();
                        }
                        for e in amplitudes_pair[RIGHT].edges.iter() {
                            ct_e_product *=
                                Into::<T>::into(2 as f64) * right_ct.onshell_edges[e.id].t.abs();
                        }
                        let common_prefactor = ct_numerator_wgt
                            * ct_e_product.inv()
                            * left_ct.cff_evaluations[LEFT][i_cff_pair[LEFT]]
                            * right_ct.cff_evaluations[RIGHT][i_cff_pair[RIGHT]];

                        let left_ct_weight = Complex::new(
                            -left_ct.e_surf_expanded.inv()
                                * left_ct.cff_pinch_dampenings[LEFT][i_cff_pair[LEFT]]
                                * left_ct.adjusted_sampling_jac
                                * left_ct.h_function_wgt,
                            if let Some(left_i_ct) = &left_ct.integrated_ct {
                                left_i_ct.e_surf_residue
                                    * left_i_ct.adjusted_sampling_jac
                                    * left_i_ct.h_function_wgt
                            } else {
                                T::zero()
                            },
                        );
                        let right_ct_weight = Complex::new(
                            -right_ct.e_surf_expanded.inv()
                                * right_ct.cff_pinch_dampenings[RIGHT][i_cff_pair[RIGHT]]
                                * right_ct.adjusted_sampling_jac
                                * right_ct.h_function_wgt,
                            // Implement the complex conjugation here with a minus sign
                            if let Some(right_i_ct) = &right_ct.integrated_ct {
                                -right_i_ct.e_surf_residue
                                    * right_i_ct.adjusted_sampling_jac
                                    * right_i_ct.h_function_wgt
                            } else {
                                T::zero()
                            },
                        );

                        let ct_im_squared_weight =
                            -left_ct_weight.im * right_ct_weight.im * common_prefactor;
                        let mut ct_weight = left_ct_weight * right_ct_weight * common_prefactor;

                        // Check if we already obtained the imaginary squared part using the trick of I - 2 AMP_CT + SG_CT,
                        // in which case we should not count it again
                        if use_imaginary_squared_trick {
                            ct_weight -= ct_im_squared_weight;
                        }

                        if use_imaginary_squared_trick
                            && !self
                                .integrand_settings
                                .threshold_ct_settings
                                .compute_only_im_squared
                        {
                            // In that case we subtracted twice the amplitude counterterms
                            ct_weight = Complex::new(
                                Into::<T>::into(2 as f64) * ct_weight.re,
                                ct_weight.im,
                            );
                        }

                        if self.settings.general.debug > 3 {
                            let amplitude_for_sides = [
                                &self.supergraph.cuts[i_cut].left_amplitude,
                                &self.supergraph.cuts[i_cut].right_amplitude,
                            ];
                            println!(
                                "   > cFF Evaluation #{} : CT for {} E-surfaces ({}) x ({}) : {:+.e} + i {:+.e}",
                                format!("{}", i_term).green(),
                                format!("L|AMP|{} x R|AMP|{}", 
                                    if left_ct.solution_type == PLUS {"+"} else {"-"},
                                    if right_ct.solution_type == PLUS {"+"} else {"-"},
                                ).purple(),
                                e_surf_str(
                                    left_ct.e_surf_id,
                                    &amplitude_for_sides[LEFT].cff_expression.e_surfaces[left_ct.e_surf_id]
                                ).blue(),
                                e_surf_str(
                                    right_ct.e_surf_id,
                                    &amplitude_for_sides[RIGHT].cff_expression.e_surfaces[right_ct.e_surf_id]
                                ).blue(),
                                ct_weight.re, ct_weight.im
                            );
                        }

                        cts_sum_for_this_term += ct_weight;
                        ct_im_squared_weight_for_this_term += ct_im_squared_weight;
                    }
                }

                cff_cts_sum += cts_sum_for_this_term;
                cff_im_squared_cts_sum += ct_im_squared_weight_for_this_term;

                if self.settings.general.debug > 2 {
                    println!(
                        "   > cFF evaluation #{} for orientation #{:-3}({}) x #{}({}):",
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
                    println!("     left  : {:+.e}", cff_left_wgt * e_product_left.inv());
                    println!("     right : {:+.e}", cff_right_wgt * e_product_right.inv());
                    println!("     num   : {:+.e}", numerator_wgt);
                    println!(
                        "{}",
                        format!("     tot   : {:+.e}", this_cff_term_contribution)
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
        for (e_id, _flip) in self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.iter() {
            e_product_cut *=
                Into::<T>::into(2 as f64) * onshell_edge_momenta_for_this_cut[*e_id].t.abs();
        }
        if self.settings.general.debug > 1 {
            println!(
                "  > On-shell energy product: {:+.e} x {:+.e} x {:+.e} = {:+.e}",
                e_product_left,
                e_product_cut,
                e_product_right,
                e_product_left * e_product_cut * e_product_right
            );
            println!(
                "  > Non-loop propagator contributions: {:+.e} x {:+.e} = {:+.e}",
                non_loop_propagators_contributions[0],
                non_loop_propagators_contributions[1],
                non_loop_propagators_contribution
            );
        }

        // Collect terms
        if self
            .integrand_settings
            .threshold_ct_settings
            .compute_only_im_squared
        {
            if self.settings.general.debug > 0 {
                println!("{}",format!("{}","   > Option 'compute_only_im_squared' enabled. Now turning off all other contributions.").red());
            }
            cff_sum = Complex::new(T::zero(), T::zero());
            // We only need to do that when trying to compute the imaginary squared part using the squaring of the integrated threshold CT.
            // if not, then we will have already done the necessary modification before.
            if !use_imaginary_squared_trick {
                cff_cts_sum = Complex::new(cff_im_squared_cts_sum, T::zero());
            } else {
                cff_cts_sum = Complex::new(cff_cts_sum.re, T::zero());
            }
        }

        cut_res = cff_sum + cff_cts_sum;
        // println!("cff_sum={:+.e}, cff_cts_sum={:+.e}", cff_sum, cff_cts_sum);

        // Collect all factors that are common for the original integrand and the threshold counterterms
        cut_res *= constants;
        cut_res *= overall_sampling_jac;
        cut_res *= t_scaling_jacobian;
        cut_res *= cut_h_function;
        cut_res *= cut_e_surface_derivative;
        cut_res *= e_product_cut.inv();
        cut_res *= non_loop_propagators_contribution;
        // println!("constants={:+.e}", constants);
        // println!("overall_sampling_jac={:+.e}", overall_sampling_jac);
        // println!("t_scaling_jacobian={:+.e}", t_scaling_jacobian);
        // println!("cut_h_function={:+.e}", cut_h_function);
        // println!("cut_e_surface_derivative={:+.e}", cut_e_surface_derivative);
        // println!("e_product_cut.inv()={:+.e}", e_product_cut);
        // println!(
        //     "non_loop_propagators_contribution={:+.e}",
        //     non_loop_propagators_contribution
        // );

        let cff_cts_sum_contribution = cff_cts_sum
            * constants
            * overall_sampling_jac
            * t_scaling_jacobian
            * cut_h_function
            * cut_e_surface_derivative
            * e_product_cut.inv();
        if self.settings.general.debug > 1 {
            println!(
            "{}",
            format!(
            "  > Result for cut #{}{} ( n_loop_left={} | cut_cardinality={} | n_loop_right={} ): {} ( ∑ CTs = {:+.e} )",
            i_cut,
            format!("({})", self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 { format!("+{}",id) } else { format!("-{}",id) }).collect::<Vec<_>>().join("|")).green(),
            self.supergraph.cuts[i_cut].left_amplitude.n_loop,
            self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.len(),
            self.supergraph.cuts[i_cut].right_amplitude.n_loop,
            format!("{:+.e}",cut_res).bold(),
            cff_cts_sum_contribution
        )
            .green()
        );
        }

        // Update the event weight to the result just computed
        self.event_manager
            .event_buffer
            .last_mut()
            .unwrap()
            .integrand = Complex::new(cut_res.re.to_f64().unwrap(), cut_res.im.to_f64().unwrap());
        return (
            cut_res,
            cff_cts_sum_contribution,
            closest_existing_e_surf,
            closest_pinched_e_surf,
        );
    }

    fn evaluate_sample_generic<T: FloatLike>(&mut self, xs: &[T]) -> Complex<T> {
        let (moms, overall_sampling_jac) = self.parameterize(xs);

        let mut loop_momenta = vec![];
        for m in &moms {
            loop_momenta.push(LorentzVector::from_args(T::zero(), m[0], m[1], m[2]));
        }
        if self.integrand_settings.sampling_basis != vec![1, 3, 6] {
            if !self.sampling_rot.is_some() {
                self.sampling_rot = Some([[0; 3]; 3]);
                let mut sig_matrix = [[0; 3]; 3];
                for i in 0..=2 {
                    for j in 0..=2 {
                        sig_matrix[i][j] = self.supergraph.edges
                            [self.integrand_settings.sampling_basis[i]]
                            .signature
                            .0[j];
                    }
                }
                self.sampling_rot = Some(utils::inv_3x3_sig_matrix(sig_matrix));
            }
            let m_rot = &self.sampling_rot.unwrap();
            let mut rotated_loop_momenta = vec![];
            for i in 0..=2 {
                let mut rotated_momenta =
                    LorentzVector::from_args(T::zero(), T::zero(), T::zero(), T::zero());
                for j in 0..=2 {
                    rotated_momenta += loop_momenta[j] * Into::<T>::into(m_rot[i][j] as f64);
                }
                rotated_loop_momenta.push(rotated_momenta);
            }
            loop_momenta = rotated_loop_momenta;
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
                    format!("{:+.e}", l.t),
                    format!("{:+.e}", l.x),
                    format!("{:+.e}", l.y),
                    format!("{:+.e}", l.z)
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
        let mut final_wgt_cts = Complex::new(T::zero(), T::zero());
        let mut overall_closest_existing_e_surf: Option<ClosestESurfaceMonitor<T>> = None;
        let mut overall_closest_pinched_e_surf: Option<ClosestESurfaceMonitor<T>> = None;
        for i_cut in 1..self.supergraph.cuts.len() {
            let (
                wgt_agg,
                wgt_cts_agg,
                closest_existing_e_surf_for_cut,
                closest_pinched_e_surf_for_cut,
            ) = self.evaluate_cut(
                i_cut,
                &loop_momenta,
                overall_sampling_jac,
                &computational_cache,
            );
            final_wgt += wgt_agg;
            final_wgt_cts += wgt_cts_agg;
            if let Some(closest) = closest_existing_e_surf_for_cut {
                if let Some(overall_closest) = &overall_closest_existing_e_surf {
                    if closest.distance.abs() < overall_closest.distance.abs() {
                        overall_closest_existing_e_surf = Some(closest.clone());
                    }
                } else {
                    overall_closest_existing_e_surf = Some(closest.clone());
                }
            }
            if let Some(closest) = closest_pinched_e_surf_for_cut {
                if let Some(overall_closest) = &overall_closest_pinched_e_surf {
                    if closest.distance.abs() < overall_closest.distance.abs() {
                        overall_closest_pinched_e_surf = Some(closest.clone());
                    }
                } else {
                    overall_closest_pinched_e_surf = Some(closest.clone());
                }
            }
        }

        if self.settings.general.debug > 0 {
            if let Some(closest) = overall_closest_existing_e_surf {
                println!(
                    "{}\n > {}\n > {:?}",
                    format!("{}", "Overall closest existing non-pinched E-surface:")
                        .blue()
                        .bold(),
                    closest.str_form(&self.supergraph.cuts[closest.i_cut]),
                    closest.e_surf_cache
                );
            }
            if let Some(closest) = overall_closest_pinched_e_surf {
                println!(
                    "{}\n > {}\n > {:?}",
                    format!("{}", "Overall closest pinched E-surface:")
                        .blue()
                        .bold(),
                    closest.str_form(&self.supergraph.cuts[closest.i_cut]),
                    closest.e_surf_cache
                );
            }
            println!(
                "{}",
                format!("total cuts weight : {:+.e}", final_wgt)
                    .green()
                    .bold()
            );
            println!(
                "{}",
                format!("( ∑ CT weights    : {:+.e} )", final_wgt_cts)
                    .green()
                    .bold()
            );
            println!(
                "{}",
                format!("Sampling jacobian : {:+.e}", overall_sampling_jac)
                    .green()
                    .bold()
            );
            println!(
                "{}",
                format!("Final contribution: {:+.e}", final_wgt)
                    .green()
                    .bold()
            );
        }
        return final_wgt;
    }
}

impl HasIntegrand for TriBoxTriIntegrand {
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

    fn get_event_manager_mut(&mut self) -> &mut EventManager {
        return &mut self.event_manager;
    }

    fn merge_results<I: HasIntegrand>(&mut self, other: &mut I, _iter: usize) {
        self.event_manager
            .merge_samples(other.get_event_manager_mut());
    }

    fn update_results(&mut self, iter: usize) {
        self.event_manager.update_result(iter);
        println!("|  -------------------------------------------------------------------------------------------");
        println!(
            "|  Fraction of rejected events : {}",
            format!(
                "{:.2}%",
                (self.event_manager.rejected_event_counter as f64)
                    / ((self.event_manager.rejected_event_counter
                        + self.event_manager.accepted_event_counter) as f64)
                    * 100.0
            )
            .blue()
        );
        // for o in &self.event_manager.observables {
        //     if let Observables::CrossSection(xs) = o {
        //         println!(
        //             "|  Cross section observable re : {}",
        //             format!("{:-19}", utils::format_uncertainty(xs.re.avg, xs.re.err)).green()
        //         );
        //         println!(
        //             "|  Cross section observable im : {}",
        //             format!("{:-19}", utils::format_uncertainty(xs.im.avg, xs.im.err)).green()
        //         );
        //     }
        // }
    }

    fn evaluate_sample(
        &mut self,
        sample: &Sample,
        wgt: f64,
        #[allow(unused)] iter: usize,
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

        let result = if use_f128 {
            let sample_xs_f128 = sample_xs
                .iter()
                .map(|x| Into::<f128::f128>::into(*x))
                .collect::<Vec<_>>();
            if self.settings.general.debug > 1 {
                println!(
                    "f128 Upcasted x-space sample : ( {} )",
                    sample_xs_f128
                        .iter()
                        .map(|&x| format!("{:+.e}", x))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            let r = self.evaluate_sample_generic(sample_xs_f128.as_slice());
            Complex::new(
                f128::f128::to_f64(&r.re).unwrap(),
                f128::f128::to_f64(&r.im).unwrap(),
            )
        } else {
            self.evaluate_sample_generic(sample_xs.as_slice())
        };
        self.event_manager.process_events(result, wgt);
        return result;
    }
}

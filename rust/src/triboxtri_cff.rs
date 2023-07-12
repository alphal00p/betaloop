use std::cmp::Ordering;

use crate::integrands::*;
#[allow(unused)]
use crate::observables::{Event, EventManager, Observables};
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
use utils::{AMPLITUDE_LEVEL_CT, LEFT, MINUS, NOSIDE, PLUS, RIGHT, SUPERGRAPH_LEVEL_CT};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TriBoxTriCFFCTSettings {
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
    pub anti_observable_settings: AntiObservableSettings,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AntiObservableSettings {
    pub enabled: bool,
    pub enable_subspace_treatment_only_when_pinches_are_closest: Option<f64>,
    pub anti_select_cut_of_subtracted_e_surface: bool,
    pub anti_select_pinched_cut_same_side_as_subtracted_e_surface: bool,
    pub choose_subspace_based_off_other_e_surface_passing_cuts: bool,
    pub use_exact_cut_selection: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TriBoxTriCFFSettings {
    pub supergraph_yaml_file: String,
    pub q: [f64; 4],
    pub h_function: HFunctionSettings,
    pub numerator: NumeratorType,
    pub sampling_basis: Vec<usize>,
    pub selected_sg_cff_term: Option<usize>,
    pub selected_cuts: Option<Vec<usize>>,
    #[serde(rename = "threshold_CT_settings")]
    pub threshold_ct_settings: TriBoxTriCFFCTSettings,
}

pub struct TriBoxTriCFFIntegrand {
    pub settings: Settings,
    pub supergraph: SuperGraph,
    pub n_dim: usize,
    pub integrand_settings: TriBoxTriCFFSettings,
    pub event_manager: EventManager,
    pub sampling_rot: Option<[[isize; 3]; 3]>,
}

pub struct TriBoxTriCFFComputationCache<T: FloatLike> {
    pub external_momenta: Vec<LorentzVector<T>>,
    pub onshell_edge_momenta_per_cut: Vec<Vec<LorentzVector<T>>>,
    pub selection_result_per_cut: Vec<bool>,
}

impl<T: FloatLike> TriBoxTriCFFComputationCache<T> {
    pub fn default() -> TriBoxTriCFFComputationCache<T> {
        TriBoxTriCFFComputationCache {
            external_momenta: vec![],
            onshell_edge_momenta_per_cut: vec![],
            selection_result_per_cut: vec![],
        }
    }
}

pub fn e_surf_str(e_surf_id: usize, e_surf: &Esurface) -> String {
    format!(
        "#{:<3} edge_ids={:-10} e_shift={:-5}",
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
    e_surf_id: usize,
    e_surf: Esurface,
    e_surf_cache: GenericESurfaceCache<T>,
}

impl<T: FloatLike + std::fmt::Debug> ClosestESurfaceMonitor<T> {
    pub fn str_form(&self, cut: &Cut) -> String {
        format!(
            "Normalised distance: {:+e} | E-surface {} of cut #{}{}: {}",
            self.distance,
            if self.e_surf_cache.get_side() == LEFT {
                "left"
            } else {
                "right"
            },
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

#[allow(unused)]
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
impl TriBoxTriCFFIntegrand {
    pub fn new(
        settings: Settings,
        integrand_settings: TriBoxTriCFFSettings,
    ) -> TriBoxTriCFFIntegrand {
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
        TriBoxTriCFFIntegrand {
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
        orientations: &Vec<(usize, isize)>,
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
                for (e_id, flip) in orientations.iter() {
                    onshell_edge_momenta_flipped[*e_id].t =
                        onshell_edge_momenta_flipped[*e_id].t.abs() * Into::<T>::into(*flip as f64);
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
        edge_ids_and_orientations: &Vec<(usize, isize)>,
        cache: &TriBoxTriCFFComputationCache<T>,
        loop_momenta: &Vec<LorentzVector<T>>,
        side: usize,
        e_shift_sig: &Vec<isize>,
    ) -> GenericESurfaceCache<T> {
        // Build the E-surface corresponding to this Cutkosky cut
        let mut ps = vec![];
        let mut sigs = vec![];
        let mut ms = vec![];
        let cut_basis_indices = self.loop_indices_in_edge_ids(esurf_basis_edge_ids);

        let mut computed_e_shift = T::zero();
        for (e_id, _flip) in edge_ids_and_orientations.iter() {
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
            //computed_e_shift -= Into::<T>::into((*flip) as f64) * shift.t;
        }
        for (i_ext, e_shift_flip) in e_shift_sig.iter().enumerate() {
            computed_e_shift +=
                cache.external_momenta[i_ext].t * Into::<T>::into((*e_shift_flip) as f64);
        }

        GenericESurfaceCache::new_from_inputs(
            cut_basis_indices,
            sigs,
            ps,
            ms,
            computed_e_shift,
            side,
            vec![],
        )
    }

    fn build_e_surfaces<T: FloatLike>(
        &self,
        onshell_edge_momenta: &Vec<LorentzVector<T>>,
        cache: &TriBoxTriCFFComputationCache<T>,
        loop_momenta: &Vec<LorentzVector<T>>,
        cff_term: &CFFTerm,
        i_cut: usize,
    ) -> Vec<GenericESurfaceCache<T>> {
        let mut e_surf_caches: Vec<GenericESurfaceCache<T>> = vec![];

        let sg_lmb = &self
            .supergraph
            .supergraph_cff
            .lmb_edges
            .iter()
            .map(|e| e.id)
            .collect::<Vec<_>>();
        let cff = &self.supergraph.supergraph_cff;

        for (i_surf, e_surf) in cff.cff_expression.e_surfaces.iter().enumerate() {
            if !cff_term.contains_e_surf_id(i_surf) {
                let mut absent_e_surf_cache = GenericESurfaceCache::default();
                absent_e_surf_cache.exists = false;
                absent_e_surf_cache.pinched = false;
                absent_e_surf_cache.eval = T::zero();
                e_surf_caches.push(absent_e_surf_cache);
                continue;
            }
            let mut e_surf_side = NOSIDE;
            for edge in &self.supergraph.cuts[i_cut].left_amplitude.edges {
                if e_surf.edge_ids.contains(&edge.id) {
                    e_surf_side = LEFT;
                    break;
                }
            }
            if e_surf_side == NOSIDE {
                for edge in &self.supergraph.cuts[i_cut].right_amplitude.edges {
                    if e_surf.edge_ids.contains(&edge.id) {
                        e_surf_side = RIGHT;
                        break;
                    }
                }
            }
            let e_surf_edge_ids_and_orientation = e_surf
                .edge_ids
                .iter()
                .map(|e_id| *cff_term.orientation.iter().find(|&o| o.0 == *e_id).unwrap())
                .collect::<Vec<_>>();
            if e_surf_side == NOSIDE {
                // For the present case we can hardcode that the tree E-surface is always pinched.
                let mut tree_e_surf_cache = self.build_e_surface_for_edges(
                    &sg_lmb,
                    &e_surf_edge_ids_and_orientation,
                    &cache,
                    // Nothing will be used from the loop momenta in this context because we specify all edges to be in the e surf basis here.
                    // But this construction is useful when building the amplitude e-surfaces using the same function.
                    loop_momenta,
                    NOSIDE,
                    &e_surf.shift,
                );
                tree_e_surf_cache.exists = false;
                tree_e_surf_cache.pinched = true;
                tree_e_surf_cache.eval = tree_e_surf_cache.eval(loop_momenta);
                e_surf_caches.push(tree_e_surf_cache);
                continue;
            }

            let amplitude_for_sides = [
                &self.supergraph.cuts[i_cut].left_amplitude,
                &self.supergraph.cuts[i_cut].right_amplitude,
            ];
            let loop_edge_ids = amplitude_for_sides[e_surf_side]
                .edges
                .iter()
                .map(|e| e.id)
                .collect::<Vec<_>>();
            let e_surf_loop_edges = e_surf
                .edge_ids
                .iter()
                .filter(|e_id| loop_edge_ids.contains(e_id))
                .map(|e_id| *e_id)
                .collect::<Vec<_>>();
            let mut e_shift_extra = T::zero();
            for e_id in &e_surf.edge_ids {
                if !e_surf_loop_edges.contains(e_id) {
                    e_shift_extra += onshell_edge_momenta[*e_id].t.abs();
                }
            }
            let e_surf_edge_ids_and_orientations = e_surf_loop_edges
                .iter()
                .map(|e_id| *cff_term.orientation.iter().find(|&o| o.0 == *e_id).unwrap())
                .collect::<Vec<_>>();
            let mut e_surf_cache = self.build_e_surface_for_edges(
                &amplitude_for_sides[e_surf_side]
                    .lmb_edges
                    .iter()
                    .map(|e| e.id)
                    .collect::<Vec<_>>(),
                &e_surf_edge_ids_and_orientations,
                &cache,
                // Nothing will be used from the loop momenta in this context
                loop_momenta,
                e_surf_side,
                &e_surf.shift,
            );
            e_surf_cache.e_shift += e_shift_extra;
            (e_surf_cache.exists, e_surf_cache.pinched) = e_surf_cache.does_exist();
            e_surf_cache.eval = e_surf_cache.eval(&loop_momenta);
            // println!(
            //     "Computing e_surf:\n{}",
            //     format!("{}", e_surf_str(i_surf, &e_surf,))
            // );
            // println!("and got\n{:?}", e_surf_cache);
            // if e_surf_cache.exists && false {
            //     e_surf_cache.t_scaling = e_surf_cache.compute_t_scaling(&loop_momenta);
            // }
            e_surf_caches.push(e_surf_cache);
        }

        e_surf_caches
    }

    fn evt_for_e_surf_to_pass_selection<T: FloatLike>(
        &self,
        asc_e_surf_id: usize,
        cache: &TriBoxTriCFFComputationCache<T>,
        onshell_edge_momenta: &Vec<LorentzVector<T>>,
    ) -> (usize, Event) {
        let (as_i_cut, anti_selected_cut) =
            match self.supergraph.cuts.iter().enumerate().find(|(_i, c)| {
                c.cut_edge_ids_and_flip.iter().all(|(e_id, _flip)| {
                    self.supergraph.supergraph_cff.cff_expression.e_surfaces[asc_e_surf_id]
                        .edge_ids
                        .contains(e_id)
                })
            }) {
                Some((i_c, c)) => (i_c, c),
                None => panic!(
                "Could not find the Cutkosky cut corresponding to this subtracted E-surface:\n{}",
                format!(
                    "{}",
                    e_surf_str(
                        asc_e_surf_id,
                        &self.supergraph.supergraph_cff.cff_expression.e_surfaces[asc_e_surf_id],
                    )
                )
            ),
            };
        let evt = self.event_manager.create_event(
            vec![cache.external_momenta[0]],
            anti_selected_cut
                .cut_edge_ids_and_flip
                .iter()
                .map(|(i_edge, flip)| onshell_edge_momenta[*i_edge] * Into::<T>::into(*flip as f64))
                .collect::<Vec<_>>(),
        );
        assert!(evt.kinematic_configuration.0.iter().all(|q| q.t > 0.));
        assert!(evt.kinematic_configuration.1.iter().all(|q| q.t > 0.));
        (as_i_cut, evt)
    }

    fn build_cts_for_e_surf_id<T: FloatLike>(
        &mut self,
        side: usize,
        e_surf_id: usize,
        scaled_loop_momenta_in_sampling_basis: &Vec<LorentzVector<T>>,
        onshell_edge_momenta_for_this_cut: &Vec<LorentzVector<T>>,
        e_surf_cache: &Vec<GenericESurfaceCache<T>>,
        cache: &TriBoxTriCFFComputationCache<T>,
        i_cut: usize,
        i_cff: usize,
        allow_subspace_projection: bool,
    ) -> Vec<ESurfaceCT<T, GenericESurfaceCache<T>>> {
        let cut = &self.supergraph.cuts[i_cut];

        let anti_observable_settings = &self
            .integrand_settings
            .threshold_ct_settings
            .anti_observable_settings;

        // Quite some gynmastic needs to take place in order to dynamically build the right basis for solving this E-surface CT
        // I semi-hard-code it for now
        // In general this is more complicated and would involve an actual change of basis, but here we can do it like this
        let mut loop_indices_for_this_ct = e_surf_cache[e_surf_id]
            .get_e_surface_basis_indices()
            .clone();

        if side == NOSIDE {
            panic!("Cannot build an E-surface CT without specifying a side.");
        }
        let other_side = if side == LEFT { RIGHT } else { LEFT };
        let amplitudes_pair = [&cut.left_amplitude, &cut.right_amplitude];
        let mut other_side_loop_indices = self.loop_indices_in_edge_ids(
            &amplitudes_pair[other_side]
                .lmb_edges
                .iter()
                .map(|e| e.id)
                .collect::<Vec<_>>(),
        );

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

        let mut edges_to_consider_when_building_subtracted_e_surf =
            self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id]
                .edge_ids
                .iter()
                .filter(|&e_id| amplitudes_pair[side].edges.iter().any(|e| e.id == *e_id))
                .map(|e_id| *e_id)
                .collect::<Vec<_>>();

        if self.settings.general.debug > 3 {
            if anti_observable_settings.enabled && !allow_subspace_projection {
                println!(
                    "{}",
                    format!("{}", "      > Subspace projection disabled by caller.").red()
                );
            }
        }

        if allow_subspace_projection
            && anti_observable_settings.enabled
            && anti_observable_settings.choose_subspace_based_off_other_e_surface_passing_cuts
        {
            if !anti_observable_settings.anti_select_cut_of_subtracted_e_surface {
                panic!("Selecting choose_subspace_based_off_other_e_surface_passing_cuts=true requires anti_select_cut_of_subtracted_e_surface=true");
            }
            // If this is a one-loop E-surface, then drop the other loop variables not part of this one-loop E-surface if there are
            // some other one-loop threshold E-surfaces with no dependence on that main one-loop E-surface variables and whose corresponding Cutkosky cut pass the selection.

            /* OLD VERSION
            // Subspace treatment same-side loop indices
            if e_surf_cache[e_surf_id].one_loop_basis_index != 99 {
                for (i_other_e_surf, other_e_surf) in e_surf_cache.iter().enumerate() {
                    if other_e_surf.side != side
                        || !other_e_surf.exists
                        || !other_e_surf.pinched
                        || other_e_surf.one_loop_basis_index == 99
                        || other_e_surf.one_loop_basis_index
                            == e_surf_cache[e_surf_id].one_loop_basis_index
                    {
                        continue;
                    }
                    let (as_i_cut, mut evt) = self.evt_for_e_surf_to_pass_selection(
                        i_other_e_surf,
                        &cache,
                        onshell_edge_momenta_for_this_cut,
                    );
                    // If this event passes the selection, it means that the corresponding other one-loop threshold CT will be disabled, so that we must solve
                    // this current one-loop CT in a hyperradius that does *not* include the degrees of freedom of that other one-loop threshold so that
                    // dual cancelations are maintained.
                    let subspace_anti_selected = self.event_manager.pass_selection(&mut evt);
                    if subspace_anti_selected {
                        loop_indices_for_this_ct.remove(
                            loop_indices_for_this_ct
                                .iter()
                                .position(|idx| *idx == other_e_surf.one_loop_basis_index)
                                .unwrap(),
                        );
                    }
                    if self.settings.general.debug > 3 {
                        println!(
                            "      > {} against cut {}{} {} remove the solving in the same-side loop index #{} of the following e-surface: {}",
                            format!("{}","Subspace anti-selection").blue(),
                            format!("#{}",as_i_cut).green(),
                            format!("({})", self.supergraph.cuts[as_i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 { format!("+{}",id) } else { format!("-{}",id) }).collect::<Vec<_>>().join("|")).blue(),
                            if subspace_anti_selected { format!("{}","DID").green() } else { format!("{}","DID NOT").red() },
                            self.supergraph.supergraph_cff.lmb_edges[other_e_surf.one_loop_basis_index].id,
                            e_surf_str(e_surf_id, &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],).red()
                        );
                    }
                }
            }
            */

            /* NEW VERSION */

            // Subspace treatment same-side loop indices
            for (i_other_e_surf, other_e_surf) in e_surf_cache.iter().enumerate() {
                let other_e_surf_loop_indices = other_e_surf.get_loop_indices_dependence();
                let loop_indices_overlap = loop_indices_for_this_ct
                    .iter()
                    .filter(|e_id| other_e_surf_loop_indices.contains(e_id))
                    .map(|e_id| *e_id)
                    .collect::<Vec<_>>();
                let edges_overlap = edges_to_consider_when_building_subtracted_e_surf
                    .iter()
                    .filter(|&e_id| {
                        self.supergraph.supergraph_cff.cff_expression.e_surfaces[i_other_e_surf]
                            .edge_ids
                            .contains(e_id)
                    })
                    .map(|e_id| *e_id)
                    .collect::<Vec<_>>();
                let sg_level_e_surf_intersection_n_edges =
                    self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id]
                        .edge_ids
                        .len()
                        + self.supergraph.supergraph_cff.cff_expression.e_surfaces[i_other_e_surf]
                            .edge_ids
                            .len()
                        - 2 * self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id]
                            .edge_ids
                            .iter()
                            .filter(|&e_id| {
                                self.supergraph.supergraph_cff.cff_expression.e_surfaces
                                    [i_other_e_surf]
                                    .edge_ids
                                    .contains(e_id)
                            })
                            .count();
                if i_other_e_surf == e_surf_id
                    || (loop_indices_overlap.len() == 0 && edges_overlap.len() == 0)
                    || (sg_level_e_surf_intersection_n_edges <= 3) // Hack to ignore anti-selection vs cut for whom this e_surf ID would be a pinched surface
                    || other_e_surf.side != side
                    || (!other_e_surf.exists && !other_e_surf.pinched)
                {
                    continue;
                }
                // println!(
                //     "This e_surf  = {} -> {:?}\n{:?}",
                //     e_surf_str(
                //         e_surf_id,
                //         &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],
                //     ),
                //     this_e_surf_loop_indices,
                //     e_surf_cache[e_surf_id]
                // );
                // println!(
                //     "other e_surf = {} -> {:?}\n{:?}",
                //     e_surf_str(
                //         i_other_e_surf,
                //         &self.supergraph.supergraph_cff.cff_expression.e_surfaces[i_other_e_surf],
                //     ),
                //     other_e_surf_loop_indices,
                //     e_surf_cache[i_other_e_surf]
                // );
                let (as_i_cut, mut evt) = self.evt_for_e_surf_to_pass_selection(
                    i_other_e_surf,
                    &cache,
                    // NOTE THAT THIS BREAKS PV BEING ZERO, something like onshell_edge_momenta_for_this_ct must be used instead
                    onshell_edge_momenta_for_this_cut,
                );
                // If this event passes the selection, it means that the corresponding other one-loop threshold CT will be disabled, so that we must solve
                // this current one-loop CT in a hyperradius that does *not* include the degrees of freedom of that other one-loop threshold so that
                // dual cancelations are maintained.
                let subspace_anti_selected = if self
                    .integrand_settings
                    .threshold_ct_settings
                    .anti_observable_settings
                    .use_exact_cut_selection
                {
                    cache.selection_result_per_cut[as_i_cut]
                } else {
                    self.event_manager.pass_selection(&mut evt)
                };
                if subspace_anti_selected {
                    for loop_index_to_remove in &loop_indices_overlap {
                        if let Some(idx_to_remove) = loop_indices_for_this_ct
                            .iter()
                            .position(|idx| idx == loop_index_to_remove)
                        {
                            loop_indices_for_this_ct.remove(idx_to_remove);
                        }
                    }
                    for edge_id_to_remove in &edges_overlap {
                        if let Some(idx_to_remove) =
                            edges_to_consider_when_building_subtracted_e_surf
                                .iter()
                                .position(|e_id| e_id == edge_id_to_remove)
                        {
                            edges_to_consider_when_building_subtracted_e_surf.remove(idx_to_remove);
                        }
                    }
                }
                if self.settings.general.debug > 3 {
                    println!(
                        "      > {} against cut {}{:-10} {} remove the solving in the same-side loop indices {} and edges {} (remaining edges after removal: {}) of the following e-surface: {}",
                        format!("{}","Subspace anti-selection").blue(),
                        format!("#{}",as_i_cut).green(),
                        format!("({})", self.supergraph.cuts[as_i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 { format!("+{}",id) } else { format!("-{}",id) }).collect::<Vec<_>>().join("|")).blue(),
                        if subspace_anti_selected { format!("{}","DID").green() } else { format!("{}","DID NOT").red() },
                        format!("({})",loop_indices_overlap.iter().map(|&li| format!("lmb#{}",li)).collect::<Vec<_>>().join(",")),
                        format!("[{}]",edges_overlap.iter().map(|&li| format!("{}",li)).collect::<Vec<_>>().join(",")),
                        format!("[{}]",edges_to_consider_when_building_subtracted_e_surf.iter().map(|&e_id| format!("#{}",e_id)).collect::<Vec<_>>().join(",")),
                        e_surf_str(e_surf_id, &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],).red()
                    );
                }
            }
            // If we need to project to a subspace of an E-surface made up of less than one square root, then ignore the counterterm altogether
            if edges_to_consider_when_building_subtracted_e_surf.len() < 2 {
                if self.settings.general.debug > 3 {
                    println!("      > {} projected away the subspace to a point so that the following e-surface {}: {}",
                        format!("{}","Subspace anti-selection").red(), 
                        format!("{}","will not be subtracted at all").red(),
                        e_surf_str(e_surf_id, &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],).red()
                    );
                }
                return vec![];
            }

            /* */

            // Subspace treatment of other-side loop indices

            // For the supergraph-level CT, we must also remove loop indices in which we solve it for all other thresholds existing on that other side
            // and sharing a degree of freedom with the one we are about to rescale
            if ct_levels_to_consider.contains(&SUPERGRAPH_LEVEL_CT) {
                for (i_other_e_surf, other_e_surf) in e_surf_cache.iter().enumerate() {
                    if other_e_surf.side != other_side
                        || (!other_e_surf.exists && !other_e_surf.pinched)
                    {
                        continue;
                    }
                    let (as_i_cut, mut evt) = self.evt_for_e_surf_to_pass_selection(
                        i_other_e_surf,
                        &cache,
                        // NOTE THAT THIS BREAKS PV BEING ZERO, something like onshell_edge_momenta_for_this_ct must be used instead
                        onshell_edge_momenta_for_this_cut,
                    );
                    // If this event passes the selection, it means that the corresponding other one-loop threshold CT will be disabled, so that we must solve
                    // this current one-loop CT in a hyperradius that does *not* include the degrees of freedom of that other one-loop threshold so that
                    // dual cancelations are maintained.
                    let subspace_anti_selected = if self
                        .integrand_settings
                        .threshold_ct_settings
                        .anti_observable_settings
                        .use_exact_cut_selection
                    {
                        cache.selection_result_per_cut[as_i_cut]
                    } else {
                        self.event_manager.pass_selection(&mut evt)
                    };
                    if subspace_anti_selected {
                        for other_e_surf_loop_index in &other_e_surf.e_surf_basis_indices {
                            if let Some(idx_to_remove) = other_side_loop_indices
                                .iter()
                                .position(|idx| idx == other_e_surf_loop_index)
                            {
                                other_side_loop_indices.remove(idx_to_remove);
                            }
                        }
                    }
                    if self.settings.general.debug > 3 {
                        println!(
                            "      > {} against cut {}{:-10} {} remove the solving in the other-side loop indices {} of the following e-surface: {}",
                            format!("{}","Subspace anti-selection").blue(),
                            format!("#{}",as_i_cut).green(),
                            format!("({})", self.supergraph.cuts[as_i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 { format!("+{}",id) } else { format!("-{}",id) }).collect::<Vec<_>>().join("|")).blue(),
                            if subspace_anti_selected { format!("{}","DID").green() } else { format!("{}","DID NOT").red() },
                            format!("({})",other_e_surf.e_surf_basis_indices.iter().map(|&li| format!("lmb#{}",li)).collect::<Vec<_>>().join(",")),
                            e_surf_str(e_surf_id, &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],).red()
                        );
                    }
                }
            }
        }

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
        let mut subtracted_e_surface = if edges_to_consider_when_building_subtracted_e_surf
            == self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id].edge_ids
        {
            e_surf_cache[e_surf_id].clone()
        } else {
            let subtracted_e_surface_edge_ids_and_orientations =
                edges_to_consider_when_building_subtracted_e_surf
                    .iter()
                    .map(|&i_e| {
                        (
                            i_e,
                            self.supergraph.supergraph_cff.cff_expression.terms[i_cff]
                                .orientation
                                .iter()
                                .find(|&o| o.0 == i_e)
                                .unwrap()
                                .1,
                        )
                    })
                    .collect::<Vec<_>>();
            let mut e_surf_to_build = self.build_e_surface_for_edges(
                &loop_indices_for_this_ct
                    .iter()
                    .map(|li| self.supergraph.supergraph_cff.lmb_edges[*li].id)
                    .collect::<Vec<_>>(),
                &subtracted_e_surface_edge_ids_and_orientations,
                &cache,
                // Nothing will be used from the loop momenta in this context because we specify all edges to be in the e surf basis here.
                // But this construction is useful when building the amplitude e-surfaces using the same function.
                scaled_loop_momenta_in_sampling_basis,
                side,
                &vec![-1, 0],
            );
            for i_e in &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id].edge_ids
            {
                if !edges_to_consider_when_building_subtracted_e_surf.contains(i_e) {
                    e_surf_to_build.e_shift += onshell_edge_momenta_for_this_cut[*i_e].t.abs();
                }
            }
            (e_surf_to_build.exists, e_surf_to_build.pinched) = e_surf_to_build.does_exist();
            if self.settings.general.debug > 3 {
                e_surf_to_build.eval = e_surf_to_build.eval(&scaled_loop_momenta_in_sampling_basis);
                println!(
                    "      > E-surface {} to subtract after projection on loop indices {} and edges {} is:\n       | {:?}",
                    e_surf_str(
                        e_surf_id,
                        &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],
                    )
                    .red(),
                    format!("({})",loop_indices_for_this_ct.iter().map(|&li| format!("lmb#{}",li)).collect::<Vec<_>>().join(",")).blue(),
                    format!("[{}]",edges_to_consider_when_building_subtracted_e_surf.iter().map(|&li| format!("{}",li)).collect::<Vec<_>>().join(",")).blue(),
                    e_surf_to_build
                );
            }
            e_surf_to_build
        };

        if !subtracted_e_surface.exists || subtracted_e_surface.pinched {
            if self.settings.general.debug > 3 {
                println!("      > The following e-surface projected into subspace with edges {} and loop indices {} {} and will therefore not be subtracted: {}",
                    format!("[{}]",edges_to_consider_when_building_subtracted_e_surf.iter().map(|&e_id| format!("#{}",e_id)).collect::<Vec<_>>().join(",")).blue(),
                    format!("({})",loop_indices_for_this_ct.iter().map(|&li| format!("lmb#{}",li)).collect::<Vec<_>>().join(",")).blue(),
                    if !subtracted_e_surface.exists {
                        format!("{}","no longer exists in that subspace").red()
                    } else {
                        format!("{}","no now pinched in that subspace").red()
                    },
                    e_surf_str(e_surf_id, &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],).red()
                );
            }
            return vec![];
        }

        let center_eval = subtracted_e_surface.eval(&center_shifts);
        assert!(center_eval < T::zero());

        // Change the parametric equation of the subtracted E-surface to the CT basis
        subtracted_e_surface.adjust_loop_momenta_shifts(&center_coordinates);

        subtracted_e_surface.t_scaling =
            subtracted_e_surface.compute_t_scaling(&loop_momenta_in_e_surf_basis);
        const _THRESHOLD: f64 = 1.0e-7;
        if subtracted_e_surface.t_scaling[MINUS] > T::zero() {
            if subtracted_e_surface.t_scaling[MINUS]
                > Into::<T>::into(_THRESHOLD * self.settings.kinematics.e_cm as f64)
            {
                panic!(
                    "Unexpected positive t-scaling for negative solution: {:+.e} for e_surf:\n{:?}",
                    subtracted_e_surface.t_scaling[MINUS], subtracted_e_surface
                );
            }
            println!("{}",format!(
                "WARNING:: Unexpected positive t-scaling for negative solution: {:+.e} for e_surf:\n{:?}",
                subtracted_e_surface.t_scaling[MINUS],subtracted_e_surface
            ).bold().red());
            subtracted_e_surface.t_scaling[MINUS] =
                -Into::<T>::into(_THRESHOLD * self.settings.kinematics.e_cm as f64);
        }
        if subtracted_e_surface.t_scaling[PLUS] < T::zero() {
            if subtracted_e_surface.t_scaling[PLUS]
                < -Into::<T>::into(_THRESHOLD * self.settings.kinematics.e_cm as f64)
            {
                panic!(
                    "Unexpected negative t-scaling for positive solution: {:+.e} for e_surf:\n{:?}",
                    subtracted_e_surface.t_scaling[PLUS], subtracted_e_surface
                );
            }
            println!("{}",format!(
                "WARNING: Unexpected negative t-scaling for positive solution: {:+.e} for e_surf:\n{:?}",
                subtracted_e_surface.t_scaling[PLUS],subtracted_e_surface).bold().red()
            );
            subtracted_e_surface.t_scaling[PLUS] =
                Into::<T>::into(_THRESHOLD * self.settings.kinematics.e_cm as f64);
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
                if self.settings.general.debug > 3 {
                    println!(
                        "      > Solving {} CT ({} solution, t={:+.16e}) in same side loop indices {} and other side loop indices {} for e-surface {}",
                        if ct_level == SUPERGRAPH_LEVEL_CT {"SG "} else {"AMP"},
                        if solution_type == PLUS {"+"} else {"-"},
                        subtracted_e_surface.t_scaling[solution_type],
                        if loop_indices_for_this_ct.len() > 0 {
                            format!("({})", loop_indices_for_this_ct.iter().map(|&i| format!("lmb#{}",i)).collect::<Vec<_>>().join(",")).blue()
                        } else {
                            format!("{}","(none)").blue()
                        },
                        if other_side_loop_indices.len() > 0 {
                            format!("({})", other_side_loop_indices.iter().map(|&i| format!("{}",self.supergraph.supergraph_cff.lmb_edges[i].id)).collect::<Vec<_>>().join("|")).blue()
                        } else {
                            format!("{}","(none)").blue()
                        },
                        e_surf_str(e_surf_id, &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],).red()
                    );
                }

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
                    if self.settings.general.debug > 4 {
                        println!(
                            "      > Counterterm didn't pass the proximity threshold, skipping it."
                        );
                    }
                    continue;
                }

                let onshell_edge_momenta_for_this_ct = self.evaluate_onshell_edge_momenta(
                    &loop_momenta_star_in_sampling_basis,
                    &cache.external_momenta,
                    &self.supergraph.supergraph_cff.cff_expression.terms[i_cff].orientation,
                );

                // Apply anti-observables
                if allow_subspace_projection && anti_observable_settings.enabled {
                    let mut anti_selected_e_surf_ids = vec![];

                    // Anti-select vs the subtracted E-surface itself, since there is anyway the Cutkosky cut contribution otherwise which will contribute to the same region.
                    if anti_observable_settings.anti_select_cut_of_subtracted_e_surface {
                        anti_selected_e_surf_ids.push(e_surf_id);
                    }

                    if anti_observable_settings
                        .anti_select_pinched_cut_same_side_as_subtracted_e_surface
                    {
                        for (other_e_surf_id, other_e_surf) in e_surf_cache.iter().enumerate() {
                            if other_e_surf_id == e_surf_id
                                || !other_e_surf.pinched
                                || other_e_surf.side != side
                            {
                                continue;
                            }
                            anti_selected_e_surf_ids.push(other_e_surf_id);
                        }
                    }

                    let mut pass_anti_selection = true;
                    for asc_e_surf_id in anti_selected_e_surf_ids {
                        let (as_i_cut, mut evt) = self.evt_for_e_surf_to_pass_selection(
                            asc_e_surf_id,
                            &cache,
                            //&onshell_edge_momenta_for_this_ct,
                            // NOTE THAT THIS BREAKS PV BEING ZERO, something like onshell_edge_momenta_for_this_ct must be used instead
                            &onshell_edge_momenta_for_this_cut,
                        );
                        let pass_selector = if self
                            .integrand_settings
                            .threshold_ct_settings
                            .anti_observable_settings
                            .use_exact_cut_selection
                        {
                            cache.selection_result_per_cut[as_i_cut]
                        } else {
                            self.event_manager.pass_selection(&mut evt)
                        };
                        if pass_selector {
                            if self.settings.general.debug > 3 {
                                println!(
                                    "      > Observable anti-selection against cut {}{} removed the subtraction of the following e-surface: {}",
                                    format!("#{}",as_i_cut).green(),
                                    format!("({})", self.supergraph.cuts[as_i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 { format!("+{}",id) } else { format!("-{}",id) }).collect::<Vec<_>>().join("|")).blue(),
                                    e_surf_str(e_surf_id, &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],).red()
                                );
                            }
                            pass_anti_selection = false;
                            break;
                        }
                    }
                    if !pass_anti_selection {
                        continue;
                    }
                }

                // Update the evaluation of the E-surface for the solved star loop momentum in the sampling basis (since it is what the shifts are computed for in this cache)
                let mut e_surface_caches_for_this_ct = [e_surf_cache.clone(), vec![]];
                for e_surf in e_surface_caches_for_this_ct[0].iter_mut() {
                    if e_surf.side != side {
                        continue;
                    }
                    e_surf.eval = e_surf.eval(&loop_momenta_star_in_sampling_basis);
                }
                if ct_level == SUPERGRAPH_LEVEL_CT {
                    for e_surf in e_surface_caches_for_this_ct[0].iter_mut() {
                        if e_surf.side != other_side {
                            continue;
                        }
                        e_surf.eval = e_surf.eval(&loop_momenta_star_in_sampling_basis);
                    }
                }

                // The factor (t - t_star) will be included at the end because we need the same quantity for the integrated CT
                // let e_surf_derivative = e_surf_cache[side][e_surf_id]
                //     .norm(&vec![
                //         loop_momenta_star_in_sampling_basis[loop_index_for_this_ct],
                //     ])
                //     .spatial_dot(&loop_momenta_star_in_e_surf_basis[loop_index_for_this_ct])
                let e_surf_derivative =
                    subtracted_e_surface.t_der(&loop_momenta_star_in_sampling_basis) / r_star;
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

                // Dummy values
                let mut cff_evaluations = [vec![], vec![]];
                let mut cff_pinch_dampenings = [vec![], vec![]];

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

                let loop_indices_solved = if side == LEFT {
                    (
                        loop_indices_for_this_ct.clone(),
                        if ct_level == SUPERGRAPH_LEVEL_CT {
                            other_side_loop_indices.clone()
                        } else {
                            vec![]
                        },
                    )
                } else {
                    (
                        if ct_level == SUPERGRAPH_LEVEL_CT {
                            other_side_loop_indices.clone()
                        } else {
                            vec![]
                        },
                        loop_indices_for_this_ct.clone(),
                    )
                };
                all_new_cts.push(ESurfaceCT {
                    e_surf_id,
                    ct_basis_signature: ct_basis_signature.clone(), // Not used at the moment, could be dropped
                    center_coordinates: center_coordinates.clone(), // Not used at the moment, could be dropped
                    adjusted_sampling_jac,
                    h_function_wgt,
                    e_surf_expanded: e_surf_expanded,
                    t: t,
                    t_star: t_star,
                    loop_momenta_star: loop_momenta_star_in_sampling_basis, // Not used at the moment, could be dropped
                    onshell_edges: onshell_edge_momenta_for_this_ct,
                    e_surface_evals: e_surface_caches_for_this_ct,
                    solution_type,        // Only for monitoring purposes
                    cff_evaluations,      // Dummy in this case
                    cff_pinch_dampenings, // Dummy in this case
                    integrated_ct,
                    ct_level,
                    loop_indices_solved,
                    ct_sector_signature: vec![], // Not used for this implementation
                    e_surface_analysis: ESurfaceCTAnalysis::default(), // Dummy in this case
                });
            }
        }
        if self.settings.general.debug > 4 {
            if all_new_cts.len() > 0 {
                println!("      > Added the following threshold counterterms:");
                println!("      =============================================");
                println!(
                    "{}",
                    all_new_cts
                        .iter()
                        .map(|ct| format!("      | {:?}", ct))
                        .collect::<Vec<_>>()
                        .join("\n      ---------------------------------------------\n")
                );
                println!("      =============================================");
            } else {
                println!("      > No threshold counterterms added.");
            }
        }
        all_new_cts
    }

    fn evaluate_onshell_edge_momenta<T: FloatLike>(
        &self,
        loop_momenta: &Vec<LorentzVector<T>>,
        external_momenta: &Vec<LorentzVector<T>>,
        orientations: &Vec<(usize, isize)>,
    ) -> Vec<LorentzVector<T>> {
        let mut onshell_edge_momenta = vec![];
        for (i, e) in self.supergraph.edges.iter().enumerate() {
            let mut edge_mom =
                utils::compute_momentum(&e.signature, &loop_momenta, &external_momenta);
            edge_mom.t = (edge_mom.spatial_squared()
                + Into::<T>::into(e.mass) * Into::<T>::into(e.mass))
            .abs()
            .sqrt();
            onshell_edge_momenta.push(edge_mom);
        }
        for (e_id, flip) in orientations.iter() {
            onshell_edge_momenta[*e_id].t *= Into::<T>::into(*flip as f64);
        }
        onshell_edge_momenta
    }

    fn compute_allow_subspace_projection_per_cff<T: FloatLike>(
        &mut self,
        loop_momenta: &Vec<LorentzVector<T>>,
        cache: &mut TriBoxTriCFFComputationCache<T>,
        selected_sg_cff_term: Option<usize>,
    ) -> Vec<bool> {
        let mut allow_subspace_projection_per_cff =
            vec![true; self.supergraph.supergraph_cff.cff_expression.terms.len()];

        if !self
            .integrand_settings
            .threshold_ct_settings
            .anti_observable_settings
            .enabled
        {
            return allow_subspace_projection_per_cff;
        }

        if self.settings.general.debug > 3 {
            println!(
                "{}",
                format!("  > Now starting global analysis of whether or not to allow subspace treatment for each cFF term").bold()
            );
        }

        cache.onshell_edge_momenta_per_cut = vec![vec![]; self.supergraph.cuts.len()];
        cache.selection_result_per_cut = vec![false; self.supergraph.cuts.len()];
        let mut info_per_active_cut = vec![];
        for i_cut in 0..self.supergraph.cuts.len() {
            if let Some(selected_cuts) = self.integrand_settings.selected_cuts.as_ref() {
                if !selected_cuts.contains(&i_cut) {
                    continue;
                }
            }
            let mut sg_cut_e_surf = 0;
            let mut e_surf_cut_found = false;
            for e_surf in self
                .supergraph
                .supergraph_cff
                .cff_expression
                .e_surfaces
                .iter()
            {
                if self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.len() == e_surf.edge_ids.len()
                    && self.supergraph.cuts[i_cut]
                        .cut_edge_ids_and_flip
                        .iter()
                        .all(|(e_id, _flip)| e_surf.edge_ids.contains(e_id))
                {
                    if e_surf.shift[1] != 0 {
                        panic!("Assumption about supergraph e-surface depending only upon the left incoming external momenta broken")
                    }
                    // Check if this e-surface is the existing one.
                    if Into::<T>::into(e_surf.shift[0] as f64) * cache.external_momenta[0].t
                        > T::zero()
                    {
                        continue;
                    }
                    sg_cut_e_surf = e_surf.id;
                    e_surf_cut_found = true;
                    break;
                }
            }
            if !e_surf_cut_found {
                panic!("Could not find the e-surface corresponding to the cut in the supergraph cff expression");
            }

            // Evaluate kinematics before forcing correct hyperradius
            let mut onshell_edge_momenta_for_this_cut = self.evaluate_onshell_edge_momenta(
                &loop_momenta,
                &cache.external_momenta,
                &self.supergraph.cuts[i_cut].cut_edge_ids_and_flip,
            );

            // Build the E-surface corresponding to this Cutkosky cut
            let mut e_surface_cc_cut = self.build_e_surface_for_edges(
                &self
                    .supergraph
                    .supergraph_cff
                    .lmb_edges
                    .iter()
                    .map(|e| e.id)
                    .collect::<Vec<_>>(),
                &self.supergraph.cuts[i_cut].cut_edge_ids_and_flip,
                &cache,
                // Nothing will be used from the loop momenta in this context because we specify all edges to be in the e surf basis here.
                // But this construction is useful when building the amplitude e-surfaces using the same function.
                loop_momenta,
                NOSIDE,
                &vec![-1, 0],
            );

            e_surface_cc_cut.t_scaling = e_surface_cc_cut.compute_t_scaling(&loop_momenta);

            let rescaled_loop_momenta = vec![
                loop_momenta[0] * e_surface_cc_cut.t_scaling[0],
                loop_momenta[1] * e_surface_cc_cut.t_scaling[0],
                loop_momenta[2] * e_surface_cc_cut.t_scaling[0],
            ];

            // Now re-evaluate the kinematics with the correct hyperradius
            onshell_edge_momenta_for_this_cut = self.evaluate_onshell_edge_momenta(
                &rescaled_loop_momenta,
                &cache.external_momenta,
                &self.supergraph.cuts[i_cut].cut_edge_ids_and_flip,
            );
            cache.onshell_edge_momenta_per_cut[i_cut] = onshell_edge_momenta_for_this_cut.clone();

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
            cache.selection_result_per_cut[i_cut] = self.event_manager.pass_selection(&mut evt);
            if !cache.selection_result_per_cut[i_cut] {
                continue;
            }

            info_per_active_cut.push((
                i_cut,
                sg_cut_e_surf,
                onshell_edge_momenta_for_this_cut,
                rescaled_loop_momenta,
            ));
        }

        if self
            .integrand_settings
            .threshold_ct_settings
            .anti_observable_settings
            .enable_subspace_treatment_only_when_pinches_are_closest
            .is_none()
        {
            return allow_subspace_projection_per_cff;
        }

        let subspace_projection_enabling_threshold = Into::<T>::into(
            self.integrand_settings
                .threshold_ct_settings
                .anti_observable_settings
                .enable_subspace_treatment_only_when_pinches_are_closest
                .unwrap(),
        );
        const _NTH_NON_PINCHED_E_SURF_TO_CONSIDER: usize = 0;
        for i_cff in 0..self.supergraph.supergraph_cff.cff_expression.terms.len() {
            if let Some(selected_term) = selected_sg_cff_term {
                if i_cff != selected_term {
                    continue;
                }
            }
            if self.settings.general.debug > 3 {
                println!("    > Now analyzing cFF term #{}", i_cff);
            }
            let mut overall_min_pinched_eval = None;
            let mut overall_min_second_non_pinched_eval = None;
            for (
                i_cut_ref,
                sg_cut_e_surf_ref,
                onshell_edge_momenta_for_this_cut,
                rescaled_loop_momenta,
            ) in info_per_active_cut.iter()
            {
                let i_cut = *i_cut_ref;
                let sg_cut_e_surf = *sg_cut_e_surf_ref;

                let mut e_surf_caches = self.build_e_surfaces(
                    &onshell_edge_momenta_for_this_cut,
                    &cache,
                    &rescaled_loop_momenta,
                    &self.supergraph.supergraph_cff.cff_expression.terms[i_cff],
                    i_cut,
                );

                if (self.supergraph.cuts[i_cut].left_amplitude.n_loop
                    + self.supergraph.cuts[i_cut].right_amplitude.n_loop)
                    >= 2
                {
                    let min_pinched_eval = e_surf_caches
                        .iter()
                        .enumerate()
                        .filter(|(i_esc, esc)| *i_esc != sg_cut_e_surf && esc.pinched)
                        .map(|(_i_esc, esc)| esc.eval.abs())
                        .min_by(|a, b| a.partial_cmp(b).unwrap());

                    if let Some(this_min_pinched_eval) = min_pinched_eval {
                        if let Some(curr_min) = overall_min_pinched_eval {
                            if curr_min > this_min_pinched_eval {
                                overall_min_pinched_eval = Some(this_min_pinched_eval);
                            }
                        } else {
                            overall_min_pinched_eval = Some(this_min_pinched_eval);
                        }
                    }
                    if self.settings.general.debug > 4 {
                        println!(
                            "    | For cut #{}, I found the following minimum pinched eval: {:?}",
                            i_cut, min_pinched_eval
                        );
                    }

                    let mut sorted_non_pinched_evals = e_surf_caches
                        .iter()
                        .enumerate()
                        .filter(|(i_esc, esc)| {
                            *i_esc != sg_cut_e_surf && !esc.pinched && esc.exists
                        })
                        .map(|(_i_esc, esc)| esc.eval.abs())
                        .collect::<Vec<_>>();
                    sorted_non_pinched_evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if sorted_non_pinched_evals.len() >= _NTH_NON_PINCHED_E_SURF_TO_CONSIDER + 1 {
                        if let Some(curr_min) = overall_min_second_non_pinched_eval {
                            if curr_min
                                > sorted_non_pinched_evals[_NTH_NON_PINCHED_E_SURF_TO_CONSIDER]
                            {
                                overall_min_second_non_pinched_eval = Some(
                                    sorted_non_pinched_evals[_NTH_NON_PINCHED_E_SURF_TO_CONSIDER],
                                );
                            }
                        } else {
                            overall_min_second_non_pinched_eval =
                                Some(sorted_non_pinched_evals[_NTH_NON_PINCHED_E_SURF_TO_CONSIDER]);
                        }
                    }
                    if self.settings.general.debug > 4 {
                        println!(
                            "    | For cut #{}, I found the following sorted non-pinched evals: {:?}",
                            i_cut, sorted_non_pinched_evals
                        );
                    }
                }
            }
            // println!("overall_min_pinched_eval={:?}", overall_min_pinched_eval);
            // println!(
            //     "overall_min_second_non_pinched_eval={:?}",
            //     overall_min_second_non_pinched_eval
            // );
            if self.settings.general.debug > 3 {
                println!(
                    "    > Determining distance to {} and {} E-surface for deciding on subspace treatment of cff term {}.",
                    format!("pinched ({})",
                        if overall_min_pinched_eval.is_some() {
                            format!("{:+.16e}",overall_min_pinched_eval.unwrap())
                        } else {
                            format!("{}","none")
                        }
                    ).red(),
                    format!("non-pinched ({})",
                        if overall_min_second_non_pinched_eval.is_some() {
                            format!("{:+.16e}",overall_min_second_non_pinched_eval.unwrap())
                        } else {
                            format!("{}","none")
                        }
                    ).green(),
                    format!("#{}",i_cff).blue()
                );
            }
            if let Some(min_pinched_eval_found) = overall_min_pinched_eval {
                if let Some(min_second_non_pinched_eval_found) = overall_min_second_non_pinched_eval
                {
                    if min_second_non_pinched_eval_found * subspace_projection_enabling_threshold
                        < min_pinched_eval_found
                    {
                        if self.settings.general.debug > 3 {
                            println!(
                                "{}",
                                format!("    > Disabling subspace projection for cff term #{}, because two non-pinched E-surfaces are closer ({:+.16e}x{:+.16e}) than the closest pinched one ({:+.16e})",
                                i_cff,min_second_non_pinched_eval_found, subspace_projection_enabling_threshold, min_pinched_eval_found
                                ).red()
                            );
                        }
                        allow_subspace_projection_per_cff[i_cff] = false;
                    }
                } else {
                    if self.settings.general.debug > 3 {
                        println!(
                            "{}",
                            format!("    > Enabling subspace projection for cff term #{} because there are no two pinched E-surfaces in any cut", i_cff).red()
                        );
                    }
                    allow_subspace_projection_per_cff[i_cff] = true;
                }
            } else {
                if self.settings.general.debug > 3 {
                    println!(
                        "{}",
                        format!("    > Disabling subspace projection for cff term #{} because there are no pinched E-surfaces in any cut", i_cff).red()
                    );
                }
                allow_subspace_projection_per_cff[i_cff] = false;
            }
            if self.settings.general.debug > 3 {
                println!(
                    "    > Outcome of the analyzing for cFF term #{}: subspace treatment {}",
                    i_cff,
                    if allow_subspace_projection_per_cff[i_cff] {
                        format!("{}", "enabled").green()
                    } else {
                        format!("{}", "disabled").red()
                    }
                );
            }
        }

        allow_subspace_projection_per_cff
    }

    fn evaluate_cut<T: FloatLike>(
        &mut self,
        i_cut: usize,
        loop_momenta: &Vec<LorentzVector<T>>,
        overall_sampling_jac: T,
        cache: &TriBoxTriCFFComputationCache<T>,
        selected_sg_cff_term: Option<usize>,
        allow_subspace_projection_per_cff: &Vec<bool>,
    ) -> (
        Complex<T>,
        Complex<T>,
        Vec<Complex<T>>,
        Vec<Complex<T>>,
        Option<ClosestESurfaceMonitor<T>>,
        Option<ClosestESurfaceMonitor<T>>,
    ) {
        let mut sg_cut_e_surf = 0;
        let mut e_surf_cut_found = false;
        for e_surf in self
            .supergraph
            .supergraph_cff
            .cff_expression
            .e_surfaces
            .iter()
        {
            if self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.len() == e_surf.edge_ids.len()
                && self.supergraph.cuts[i_cut]
                    .cut_edge_ids_and_flip
                    .iter()
                    .all(|(e_id, _flip)| e_surf.edge_ids.contains(e_id))
            {
                if e_surf.shift[1] != 0 {
                    panic!("Assumption about supergraph e-surface depending only upon the left incoming external momenta broken")
                }
                // Check if this e-surface is the existing one.
                if Into::<T>::into(e_surf.shift[0] as f64) * cache.external_momenta[0].t > T::zero()
                {
                    continue;
                }
                sg_cut_e_surf = e_surf.id;
                e_surf_cut_found = true;
                break;
            }
        }
        if !e_surf_cut_found {
            panic!("Could not find the e-surface corresponding to the cut in the supergraph cff expression");
        }

        let mut cut_res = Complex::new(T::zero(), T::zero());
        if self.settings.general.debug > 1 {
            println!(
                "{}",
                format!(
                "  > Starting evaluation of cut #{}{} ( sg_e_surf#={} | n_loop_left={} | cut_cardinality={} | n_loop_right={} )",
                i_cut,
                format!("({})", self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 { format!("+{}",id) } else { format!("-{}",id) }).collect::<Vec<_>>().join("|")).blue(),
                sg_cut_e_surf,
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
        let mut onshell_edge_momenta_for_this_cut = self.evaluate_onshell_edge_momenta(
            &loop_momenta,
            &cache.external_momenta,
            &self.supergraph.cuts[i_cut].cut_edge_ids_and_flip,
        );

        // Build the E-surface corresponding to this Cutkosky cut
        let mut e_surface_cc_cut = self.build_e_surface_for_edges(
            &self
                .supergraph
                .supergraph_cff
                .lmb_edges
                .iter()
                .map(|e| e.id)
                .collect::<Vec<_>>(),
            &self.supergraph.cuts[i_cut].cut_edge_ids_and_flip,
            &cache,
            // Nothing will be used from the loop momenta in this context because we specify all edges to be in the e surf basis here.
            // But this construction is useful when building the amplitude e-surfaces using the same function.
            loop_momenta,
            NOSIDE,
            &vec![-1, 0],
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
        onshell_edge_momenta_for_this_cut = self.evaluate_onshell_edge_momenta(
            &rescaled_loop_momenta,
            &cache.external_momenta,
            &self.supergraph.cuts[i_cut].cut_edge_ids_and_flip,
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
            if self.settings.general.debug > 1 {
                println!(
                    "{}",
                    format!(
                        "    The following event for cut #{}{} failed to pass the cuts:\n    {:?}",
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
                    )
                    .red()
                );
            }
            return (
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::zero(), T::zero()),
                vec![
                    Complex::new(T::zero(), T::zero());
                    self.supergraph.supergraph_cff.cff_expression.terms.len()
                ],
                vec![
                    Complex::new(T::zero(), T::zero());
                    self.supergraph.supergraph_cff.cff_expression.terms.len()
                ],
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

        let mut closest_existing_e_surf: Option<ClosestESurfaceMonitor<T>> = None;
        let mut closest_pinched_e_surf: Option<ClosestESurfaceMonitor<T>> = None;

        let mut e_product = T::one();
        for e_id in 0..self.supergraph.edges.len() {
            e_product *=
                Into::<T>::into(2 as f64) * onshell_edge_momenta_for_this_cut[e_id].t.abs();
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
        let mut cff_res = vec![
            Complex::new(T::zero(), T::zero());
            self.supergraph.supergraph_cff.cff_expression.terms.len()
        ];
        let mut cff_cts_res = vec![
            Complex::new(T::zero(), T::zero());
            self.supergraph.supergraph_cff.cff_expression.terms.len()
        ];
        let mut cff_im_squared_cts_res =
            vec![T::zero(); self.supergraph.supergraph_cff.cff_expression.terms.len()];

        for i_cff in 0..self.supergraph.supergraph_cff.cff_expression.terms.len() {
            if let Some(selected_term) = selected_sg_cff_term {
                if i_cff != selected_term {
                    continue;
                }
            }

            // Adjust the energy signs for this cff term
            for (e_id, flip) in self.supergraph.supergraph_cff.cff_expression.terms[i_cff]
                .orientation
                .iter()
            {
                onshell_edge_momenta_for_this_cut[*e_id].t =
                    onshell_edge_momenta_for_this_cut[*e_id].t.abs()
                        * Into::<T>::into(*flip as f64);
            }

            if !self.supergraph.supergraph_cff.cff_expression.terms[i_cff]
                .contains_e_surf_id(sg_cut_e_surf)
            {
                if self.settings.general.debug > 2 {
                    println!(
                        "   > SG cFF evaluation for orientation #{:-3}({}): {}",
                        format!("{}", i_cff).red(),
                        self.supergraph.supergraph_cff.cff_expression.terms[i_cff]
                            .orientation
                            .iter()
                            .map(|(_id, flip)| if *flip > 0 { "+" } else { "-" })
                            .collect::<Vec<_>>()
                            .join("")
                            .blue(),
                        format!("Cutkosky cut absent in this term.").red()
                    );
                }
                continue;
            }
            if self.settings.general.debug > 2 {
                println!(
                    "   > SG cFF evaluation for orientation #{:-3}({}): {}",
                    format!("{}", i_cff).green(),
                    self.supergraph.supergraph_cff.cff_expression.terms[i_cff]
                        .orientation
                        .iter()
                        .map(|(_id, flip)| if *flip > 0 { "+" } else { "-" })
                        .collect::<Vec<_>>()
                        .join("")
                        .blue(),
                    format!("Contains Cutkosky cut. Starting its computation now.").green()
                );
            }
            // Now build and evaluate SG E-surfaces for this cut, projected onto the on-shell cut edges
            let mut e_surf_caches: Vec<GenericESurfaceCache<T>> = self.build_e_surfaces(
                &onshell_edge_momenta_for_this_cut,
                &cache,
                &rescaled_loop_momenta,
                &self.supergraph.supergraph_cff.cff_expression.terms[i_cff],
                i_cut,
            );

            if self.settings.general.debug > 0 {
                for (i_surf, e_surf_cache) in e_surf_caches.iter().enumerate() {
                    if i_surf == sg_cut_e_surf {
                        continue;
                    }
                    let new_monitor = ClosestESurfaceMonitor {
                        distance: e_surf_cache.eval
                            / Into::<T>::into(self.settings.kinematics.e_cm as f64),
                        e_surf_id: i_surf,
                        i_cut: i_cut,
                        e_surf: self.supergraph.supergraph_cff.cff_expression.e_surfaces[i_surf]
                            .clone(),
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
            if self.settings.general.debug > 3 {
                let amplitude_for_sides = [
                    &self.supergraph.cuts[i_cut].left_amplitude,
                    &self.supergraph.cuts[i_cut].right_amplitude,
                ];
                if e_surf_caches.len() == 0 {
                    println!(
                        "    | All e_surf caches:      {}",
                        format!("{}", "None").green()
                    );
                } else {
                    let cff_term = &self.supergraph.supergraph_cff.cff_expression.terms[i_cff];
                    let mut n_e_surf_to_subtract = 0;
                    let e_surfs_for_this_cff = e_surf_caches
                        .iter()
                        .enumerate()
                        .filter(|(i_esc, _esc)| cff_term.contains_e_surf_id(*i_esc))
                        .map(|(_i_esc, esc)| esc)
                        .collect::<Vec<_>>();
                    for es in e_surfs_for_this_cff {
                        if es.exists && !es.pinched {
                            n_e_surf_to_subtract += 1;
                        }
                    }
                    println!(
                        "    All e_surf caches {}:\n      {}",
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
                        e_surf_caches
                            .iter()
                            .enumerate()
                            .filter(|(i_e, _e)| cff_term.contains_e_surf_id(*i_e))
                            .map(|(i_esurf, es)| format!(
                                "{} : {:?}",
                                if es.exists && !es.pinched {
                                    e_surf_str(
                                        i_esurf,
                                        &self.supergraph.supergraph_cff.cff_expression.e_surfaces
                                            [i_esurf],
                                    )
                                    .bold()
                                    .red()
                                } else {
                                    e_surf_str(
                                        i_esurf,
                                        &self.supergraph.supergraph_cff.cff_expression.e_surfaces
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

            // Now build the counterterms
            let mut cts = [vec![], vec![]];
            if self.integrand_settings.threshold_ct_settings.enabled {
                // There are smarter ways to do this, but this is the most straightforward and clear for this exploration
                for side in [LEFT, RIGHT] {
                    for e_surf_id in 0..self
                        .supergraph
                        .supergraph_cff
                        .cff_expression
                        .e_surfaces
                        .len()
                    {
                        if e_surf_caches[e_surf_id].side != side {
                            continue;
                        }
                        if e_surf_caches[e_surf_id].exists && !e_surf_caches[e_surf_id].pinched {
                            if self.settings.general.debug > 3 {
                                let amplitude_for_sides = [
                                    &self.supergraph.cuts[i_cut].left_amplitude,
                                    &self.supergraph.cuts[i_cut].right_amplitude,
                                ];
                                println!(
                                    "    | Now subtracting E-surface {}",
                                    format!(
                                        "{} : {:?}",
                                        e_surf_str(
                                            e_surf_id,
                                            &self
                                                .supergraph
                                                .supergraph_cff
                                                .cff_expression
                                                .e_surfaces[e_surf_id]
                                        )
                                        .bold()
                                        .red(),
                                        e_surf_caches[e_surf_id]
                                    )
                                );
                            }
                            let new_cts = self.build_cts_for_e_surf_id(
                                side,
                                e_surf_id,
                                &rescaled_loop_momenta,
                                &onshell_edge_momenta_for_this_cut,
                                &e_surf_caches,
                                cache,
                                i_cut,
                                i_cff,
                                allow_subspace_projection_per_cff[i_cff],
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

            let mut numerator_wgt = self.evaluate_numerator(
                &onshell_edge_momenta_for_this_cut,
                &self.supergraph.supergraph_cff.cff_expression.terms[i_cff].orientation,
            );

            let cff_eval = self.supergraph.supergraph_cff.cff_expression.terms[i_cff]
                .evaluate(&e_surf_caches, Some(vec![(sg_cut_e_surf, T::one())]));

            let mut this_cff_term_contribution = numerator_wgt * cff_eval * e_product.inv();

            if use_imaginary_squared_trick
                && self
                    .integrand_settings
                    .threshold_ct_settings
                    .compute_only_im_squared
            {
                this_cff_term_contribution = T::zero();
            }
            cff_res[i_cff] += this_cff_term_contribution;

            // Now include counterterms
            let mut cts_sum_for_this_term = Complex::new(T::zero(), T::zero());
            for ct_side in [LEFT, RIGHT] {
                let other_side = if ct_side == LEFT { RIGHT } else { LEFT };
                for ct in cts[ct_side].iter() {
                    let cff_term = &self.supergraph.supergraph_cff.cff_expression.terms[i_cff];
                    let ct_numerator_wgt =
                        self.evaluate_numerator(&ct.onshell_edges, &cff_term.orientation);

                    let mut ct_e_product = T::one();
                    for e in self.supergraph.edges.iter() {
                        ct_e_product *= Into::<T>::into(2 as f64) * ct.onshell_edges[e.id].t.abs();
                    }

                    let ct_cff_eval = cff_term.evaluate(
                        &ct.e_surface_evals[0],
                        Some(vec![(sg_cut_e_surf, T::one()), (ct.e_surf_id, T::one())]),
                    );

                    let mut re_ct_weight = -ct_numerator_wgt
                        * ct_e_product.inv()
                        * ct_cff_eval
                        * ct.e_surf_expanded.inv()
                        * ct.adjusted_sampling_jac
                        * ct.h_function_wgt;

                    let mut im_ct_weight = if let Some(i_ct) = &ct.integrated_ct {
                        ct_numerator_wgt
                            * ct_e_product.inv()
                            * ct_cff_eval
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
                    //println!("CT={:?}",ct);
                    if self.settings.general.debug > 3 {
                        println!(
                            "   > cFF Evaluation #{} : CT for {} E-surface {} solved in {}: {:+.e} + i {:+.e}",
                            format!("{}", i_cff).green(),
                            format!(
                                "{}|{}|{}",
                                if ct_side == LEFT { "L" } else { "R" },
                                if ct.ct_level == AMPLITUDE_LEVEL_CT {
                                    "AMP"
                                } else {
                                    "SG "
                                },
                                if ct.solution_type == PLUS { "+" } else { "-" }
                            )
                            .purple(),
                            e_surf_str(
                                ct.e_surf_id,
                                &self.supergraph.supergraph_cff.cff_expression.e_surfaces
                                    [ct.e_surf_id]
                            )
                            .blue(),
                            format!("[{}x{}]",
                                format!("({:-3})", ct.loop_indices_solved.0.iter().map(|lis| format!("{}", lis)).collect::<Vec<_>>().join(",")),
                                format!("({:-3})", ct.loop_indices_solved.1.iter().map(|lis| format!("{}", lis)).collect::<Vec<_>>().join(",")),
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
                if left_ct.ct_level == SUPERGRAPH_LEVEL_CT {
                    continue;
                }
                for right_ct in cts[RIGHT].iter() {
                    if right_ct.ct_level == SUPERGRAPH_LEVEL_CT {
                        continue;
                    }
                    let cff_term = &self.supergraph.supergraph_cff.cff_expression.terms[i_cff];
                    let amplitudes_pair = [
                        &self.supergraph.cuts[i_cut].left_amplitude,
                        &self.supergraph.cuts[i_cut].right_amplitude,
                    ];

                    let mut combined_onshell_edges = left_ct.onshell_edges.clone();
                    for edge in &amplitudes_pair[RIGHT].edges {
                        combined_onshell_edges[edge.id] = right_ct.onshell_edges[edge.id];
                    }
                    let ct_numerator_wgt =
                        self.evaluate_numerator(&combined_onshell_edges, &cff_term.orientation);
                    let mut ct_e_product = T::one();
                    for e in self.supergraph.edges.iter() {
                        ct_e_product *=
                            Into::<T>::into(2 as f64) * combined_onshell_edges[e.id].t.abs();
                    }

                    let mut combined_e_surface_caches = left_ct.e_surface_evals[0].clone();
                    for (i_surf, e_surf_cache) in right_ct.e_surface_evals[0].iter().enumerate() {
                        if e_surf_cache.side == RIGHT {
                            combined_e_surface_caches[i_surf] = e_surf_cache.clone();
                        }
                    }

                    let ct_cff_eval = cff_term.evaluate(
                        &combined_e_surface_caches,
                        Some(vec![
                            (sg_cut_e_surf, T::one()),
                            (left_ct.e_surf_id, T::one()),
                            (right_ct.e_surf_id, T::one()),
                        ]),
                    );

                    let common_prefactor = ct_numerator_wgt * ct_e_product.inv() * ct_cff_eval;

                    let left_ct_weight = Complex::new(
                        -left_ct.e_surf_expanded.inv()
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
                        ct_weight =
                            Complex::new(Into::<T>::into(2 as f64) * ct_weight.re, ct_weight.im);
                    }

                    if self.settings.general.debug > 3 {
                        println!(
                                "   > cFF Evaluation #{} : CT for {} E-surfaces ({}) x ({}) : {:+.e} + i {:+.e}",
                                format!("{}", i_cff).green(),
                                format!("L|AMP|{} x R|AMP|{}", 
                                    if left_ct.solution_type == PLUS {"+"} else {"-"},
                                    if right_ct.solution_type == PLUS {"+"} else {"-"},
                                ).purple(),
                                e_surf_str(
                                    left_ct.e_surf_id,
                                    &self.supergraph.supergraph_cff.cff_expression.e_surfaces[left_ct.e_surf_id]
                                ).blue(),
                                e_surf_str(
                                    right_ct.e_surf_id,
                                    &self.supergraph.supergraph_cff.cff_expression.e_surfaces[right_ct.e_surf_id]
                                ).blue(),
                                ct_weight.re, ct_weight.im
                            );
                    }

                    cts_sum_for_this_term += ct_weight;
                    ct_im_squared_weight_for_this_term += ct_im_squared_weight;
                }
            }

            cff_cts_res[i_cff] += cts_sum_for_this_term;
            cff_im_squared_cts_res[i_cff] += ct_im_squared_weight_for_this_term;

            if self.settings.general.debug > 2 {
                let cff_term = &self.supergraph.supergraph_cff.cff_expression.terms[i_cff];
                println!(
                    "   > cFF evaluation #{} for orientation ({}):",
                    format!("{}", i_cff).green(),
                    cff_term
                        .orientation
                        .iter()
                        .map(|(_id, flip)| if *flip > 0 { "+" } else { "-" })
                        .collect::<Vec<_>>()
                        .join("")
                        .blue()
                );
                println!("     eprod : {:+.e}", e_product.inv());
                println!("     cff   : {:+.e}", cff_eval);
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

        // Collect terms
        if self
            .integrand_settings
            .threshold_ct_settings
            .compute_only_im_squared
        {
            if self.settings.general.debug > 0 {
                println!("{}",format!("{}","   > Option 'compute_only_im_squared' enabled. Now turning off all other contributions.").red());
            }
            for i_cff in 0..self.supergraph.supergraph_cff.cff_expression.terms.len() {
                cff_res[i_cff] = Complex::new(T::zero(), T::zero());
                // We only need to do that when trying to compute the imaginary squared part using the squaring of the integrated threshold CT.
                // if not, then we will have already done the necessary modification before.
                if !use_imaginary_squared_trick {
                    cff_cts_res[i_cff] = Complex::new(cff_im_squared_cts_res[i_cff], T::zero());
                } else {
                    cff_cts_res[i_cff] = Complex::new(cff_cts_res[i_cff].re, T::zero());
                }
            }
        }

        if self.settings.general.debug > 2 && selected_sg_cff_term.is_none() {
            let mut all_res = cff_res
                .iter()
                .enumerate()
                .zip(&cff_cts_res)
                .map(|((i_a_cff, a), a_ct)| (i_a_cff, *a - *a_ct, *a_ct))
                .collect::<Vec<_>>();
            all_res.sort_by(|(i_b_cff, b, b_ct), (i_a_cff, a, a_ct)| {
                (a + a_ct)
                    .re
                    .abs()
                    .partial_cmp(&(b + b_ct).re.abs())
                    .unwrap_or(Ordering::Equal)
            });
            let sorted_cff_res = all_res
                .iter()
                .filter(|(i_cff, res_no_ct, ct)| (res_no_ct + ct).re.abs() > T::zero())
                .map(|(i_cff, res_no_ct, ct)| {
                    format!(
                        "  | #{:-3} ({}) : {:-23} ( {:-23} | {:+.e} )",
                        format!("{}", i_cff).green(),
                        self.supergraph.supergraph_cff.cff_expression.terms[*i_cff]
                            .orientation
                            .iter()
                            .map(|(_id, flip)| if *flip > 0 { "+" } else { "-" })
                            .collect::<Vec<_>>()
                            .join("")
                            .blue(),
                        format!("{:+.e}", (res_no_ct + ct).re).bold(),
                        format!("{:+.e}", res_no_ct.re),
                        ct.re
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            let n_non_zero = all_res
                .iter()
                .filter(|(i_cff, res_no_ct, ct)| (res_no_ct + ct).re.abs() > T::zero())
                .count();
            if n_non_zero > 0 {
                println!("{}",format!("  > All {} sorted non-zero real results for independent cFF terms for cut #{}{} ( n_loop_left={} | cut_cardinality={} | n_loop_right={} ):",
                n_non_zero,
                i_cut,
                format!("({})", self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 { format!("+{}",id) } else { format!("-{}",id) }).collect::<Vec<_>>().join("|")).green(),
                self.supergraph.cuts[i_cut].left_amplitude.n_loop,
                self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.len(),
                self.supergraph.cuts[i_cut].right_amplitude.n_loop
            ).green());
                println!(
                    "{}",
                    format!(
                        "  | #{:-3} ({}) : {:-23} ( {:-23} | {} )",
                        format!("{}", "ID").green(),
                        format!("{}", "12345678").blue(),
                        format!("{}", "res + CTs").bold(),
                        format!("{}", "no CTs"),
                        format!("{}", "∑ CTs")
                    ),
                );
                println!("{}", sorted_cff_res);
            }
        }

        let cff_sum = cff_res.iter().sum::<Complex<T>>();
        let cff_cts_sum = cff_cts_res.iter().sum::<Complex<T>>();
        let cff_im_squared_cts_sum = cff_im_squared_cts_res.iter().map(|r| *r).sum::<T>();

        cut_res = cff_sum + cff_cts_sum;
        // println!("cff_sum={:+.e}, cff_cts_sum={:+.e}", cff_sum, cff_cts_sum);

        let global_factors = constants
            * overall_sampling_jac
            * t_scaling_jacobian
            * cut_h_function
            * cut_e_surface_derivative;
        // println!("constants={:+.e}", constants);
        // println!("overall_sampling_jac={:+.e}", overall_sampling_jac);
        // println!("t_scaling_jacobian={:+.e}", t_scaling_jacobian);
        // println!("cut_h_function={:+.e}", cut_h_function);
        // println!("cut_e_surface_derivative={:+.e}", cut_e_surface_derivative);

        // Collect all factors that are common for the original integrand and the threshold counterterms
        cut_res *= global_factors;

        let cff_cts_sum_contribution = cff_cts_sum * global_factors;
        for i_cff in 0..self.supergraph.supergraph_cff.cff_expression.terms.len() {
            cff_cts_res[i_cff] *= global_factors;
            cff_res[i_cff] *= global_factors;
        }

        if self.settings.general.debug > 1 {
            println!(
            "{}",
            format!(
            "  > Result for cut #{}{} ( n_loop_left={} | cut_cardinality={} | n_loop_right={} ): {} ( no CTs = {:+.e} | ∑ CTs = {:+.e} )",
            i_cut,
            format!("({})", self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 { format!("+{}",id) } else { format!("-{}",id) }).collect::<Vec<_>>().join("|")).green(),
            self.supergraph.cuts[i_cut].left_amplitude.n_loop,
            self.supergraph.cuts[i_cut].cut_edge_ids_and_flip.len(),
            self.supergraph.cuts[i_cut].right_amplitude.n_loop,
            format!("{:+.e}",cut_res).bold(),
            cut_res-cff_cts_sum_contribution,
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
            cff_res,
            cff_cts_res,
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

        let mut computational_cache = TriBoxTriCFFComputationCache::default();

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
        let mut final_wgt_per_cff = vec![
            Complex::new(T::zero(), T::zero());
            self.supergraph.supergraph_cff.cff_expression.terms.len()
        ];
        let mut final_wgt_per_cff_cts =
            vec![
                Complex::new(T::zero(), T::zero());
                self.supergraph.supergraph_cff.cff_expression.terms.len()
            ];
        let mut final_wgt_per_cut =
            vec![Complex::new(T::zero(), T::zero()); self.supergraph.cuts.len()];
        let mut final_wgt_cts_per_cut =
            vec![Complex::new(T::zero(), T::zero()); self.supergraph.cuts.len()];

        let allow_subspace_projection_per_cff: Vec<bool> = self
            .compute_allow_subspace_projection_per_cff(
                &loop_momenta,
                &mut computational_cache,
                self.integrand_settings.selected_sg_cff_term,
            );

        for i_cut in 0..self.supergraph.cuts.len() {
            if let Some(selected_cuts) = self.integrand_settings.selected_cuts.as_ref() {
                if !selected_cuts.contains(&i_cut) {
                    continue;
                }
            }
            let (
                wgt_agg,
                wgt_cts_agg,
                wgt_agg_per_cff,
                wgt_cts_agg_per_cff,
                closest_existing_e_surf_for_cut,
                closest_pinched_e_surf_for_cut,
            ) = self.evaluate_cut(
                i_cut,
                &loop_momenta,
                overall_sampling_jac,
                &computational_cache,
                self.integrand_settings.selected_sg_cff_term,
                &allow_subspace_projection_per_cff,
            );
            final_wgt += wgt_agg;
            final_wgt_cts += wgt_cts_agg;
            final_wgt_per_cut[i_cut] += wgt_agg;
            final_wgt_cts_per_cut[i_cut] += wgt_cts_agg;
            for i_cff in 0..self.supergraph.supergraph_cff.cff_expression.terms.len() {
                final_wgt_per_cff[i_cff] += wgt_agg_per_cff[i_cff];
                final_wgt_per_cff_cts[i_cff] += wgt_cts_agg_per_cff[i_cff];
            }
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

        if self.settings.general.debug > 2 {
            let mut all_res = final_wgt_per_cff
                .iter()
                .enumerate()
                .zip(&final_wgt_per_cff_cts)
                .map(|((i_a_cff, a), a_ct)| (i_a_cff, *a - *a_ct, *a_ct))
                .collect::<Vec<_>>();
            all_res.sort_by(|(i_b_cff, b, b_ct), (i_a_cff, a, a_ct)| {
                (a + a_ct)
                    .re
                    .abs()
                    .partial_cmp(&(b + b_ct).re.abs())
                    .unwrap_or(Ordering::Equal)
            });
            let n_non_zero = all_res
                .iter()
                .filter(|(i_cff, res_no_ct, ct)| (res_no_ct + ct).re.abs() > T::zero())
                .count();
            let sorted_cff_res = all_res
                .iter()
                .filter(|(i_cff, res_no_ct, ct)| (res_no_ct + ct).re.abs() > T::zero())
                .map(|(i_cff, res_no_ct, ct)| {
                    format!(
                        "  | #{:-3} ({})     : {:-23} ( {:-23} | {:+.e} )",
                        format!("{}", i_cff).green(),
                        self.supergraph.supergraph_cff.cff_expression.terms[*i_cff]
                            .orientation
                            .iter()
                            .map(|(_id, flip)| if *flip > 0 { "+" } else { "-" })
                            .collect::<Vec<_>>()
                            .join("")
                            .blue(),
                        format!("{:+.e}", (res_no_ct + ct).re).bold(),
                        format!("{:+.e}", res_no_ct.re),
                        ct.re
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            if n_non_zero > 0 {
                println!("{}",
                format!("  > All {} sorted non-zero real results for independent cFF terms for the sum over all cuts:",n_non_zero).green().bold()
            );
                println!(
                    "{}",
                    format!(
                        "  | #{:-3} ({})     : {:-23} ( {:-23} | {} )",
                        format!("{}", "ID").green(),
                        format!("{}", "12345678").blue(),
                        format!("{}", "res + CTs").bold(),
                        format!("{}", "no CTs"),
                        format!("{}", "∑ CTs")
                    ),
                );
                println!("{}", sorted_cff_res);
            }
        }

        if self.settings.general.debug > 1 {
            let mut all_res = final_wgt_per_cut
                .iter()
                .enumerate()
                .zip(&final_wgt_cts_per_cut)
                .map(|((i_a_cut, a), a_ct)| (i_a_cut, *a - *a_ct, *a_ct))
                .collect::<Vec<_>>();

            let sorted_cut_res = all_res
                .iter()
                .map(|(i_cut, res_no_ct, ct)| {
                    format!(
                        "  | #{:-3} ({:-12}) : {:-23} ( {:-23} | {:+.e} )",
                        format!("{}", i_cut).green(),
                        format!(
                            "{}",
                            self.supergraph.cuts[*i_cut]
                                .cut_edge_ids_and_flip
                                .iter()
                                .map(|(id, flip)| if *flip > 0 {
                                    format!("+{}", id)
                                } else {
                                    format!("-{}", id)
                                })
                                .collect::<Vec<_>>()
                                .join("|")
                        )
                        .blue(),
                        format!("{:+.e}", (res_no_ct + ct).re).bold(),
                        format!("{:+.e}", res_no_ct.re),
                        ct.re
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            println!(
                "{}",
                format!(
                    "  > All {} real results for each Cutkosky cut:",
                    all_res.len()
                )
                .green()
                .bold()
            );
            println!(
                "{}",
                format!(
                    "  | {:-4} ({:-12}) : {:-23} ( {:-23} | {} )",
                    format!("{}", "ID").green(),
                    format!("{}", "cut edges").blue(),
                    format!("{}", "res + CTs").bold(),
                    format!("{}", "no CTs"),
                    format!("{}", "∑ CTs")
                ),
            );
            println!("  ----------------------------------------------------------------------------------------------------");
            println!("{}", sorted_cut_res);
            println!("  ----------------------------------------------------------------------------------------------------");
            println!(
                "{}",
                format!(
                    "  | {:-4} ({:-12}) : {:-23} ( {:-23} | {:+.e} )",
                    format!("{}", "TOT").green(),
                    format!("{}", "∑ cuts").blue(),
                    format!("{:+.e}", all_res.iter().map(|r| (r.1 + r.2).re).sum::<T>()).bold(),
                    format!("{:+.e}", all_res.iter().map(|r| r.1.re).sum::<T>()),
                    all_res.iter().map(|r| r.2.re).sum::<T>()
                ),
            );
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

impl HasIntegrand for TriBoxTriCFFIntegrand {
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

        let mut result = if use_f128 {
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
        if !result.re.is_finite() || !result.im.is_finite() {
            println!("{}",format!("WARNING: Found infinite result (now set to zero) for xs={:?} : {:+16e} + i {:+16e}", sample_xs, result.re, result.im).bold().red());
            result = Complex::new(0.0, 0.0);
        }
        return result;
    }
}

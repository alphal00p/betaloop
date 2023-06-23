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
use std::fs::File;
use utils::{
    AMPLITUDE_LEVEL_CT, CUT_ABSENT, CUT_ACTIVE, CUT_INACTIVE, LEFT, MINUS, NOSIDE, PLUS, RIGHT,
    SUPERGRAPH_LEVEL_CT,
};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TriBoxTriCFFSectoredCTSettings {
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
    pub include_cts_solved_in_two_loop_space: bool,
    pub include_cts_solved_in_one_loop_subspace: bool,
    pub sectoring_settings: SectoringSettings,
    pub apply_original_event_selection_to_cts: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SectoringSettings {
    pub enabled: bool,
    pub accept_all: bool,
    pub sector_based_analysis: bool,
    pub force_one_loop_ct_in_soft_sector: bool,
    pub always_solve_cts_in_all_amplitude_loop_indices: bool,
    pub anti_select_threshold_against_observable: bool,
    pub correlate_event_sector_with_ct_sector: bool,
    pub apply_hard_coded_rules: bool,
    pub check_for_absent_e_surfaces_when_building_mc_factor: bool,
    pub mc_factor_power: f64, // negative means using min()
    pub hard_coded_rules_file: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SectoringRule {
    pub sector_signature: Vec<isize>,
    pub rules_for_cut: Vec<SectoringRuleForCut>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SectoringRuleForCut {
    pub cut_id: usize,
    pub rules_for_ct: Vec<SectoringRuleForCutAndCT>,
}

fn _default_mc_factor() -> SectoringRuleMCFactor {
    SectoringRuleMCFactor {
        e_surf_ids_prod_in_num: vec![],
        e_surf_ids_prods_to_sum_in_denom: vec![],
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SectoringRuleForCutAndCT {
    pub surf_id_subtracted: usize,
    pub loop_indices_this_ct_is_solved_in: Vec<usize>,
    pub enabled: bool,
    #[serde(default = "_default_mc_factor")]
    pub mc_factor: SectoringRuleMCFactor,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SectoringRuleMCFactor {
    pub e_surf_ids_prod_in_num: Vec<(usize, usize)>,
    pub e_surf_ids_prods_to_sum_in_denom: Vec<Vec<(usize, usize)>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TriBoxTriCFFSectoredSettings {
    pub supergraph_yaml_file: String,
    pub q: [f64; 4],
    pub h_function: HFunctionSettings,
    pub numerator: NumeratorType,
    pub sampling_basis: Vec<usize>,
    pub selected_sg_cff_term: Option<usize>,
    pub selected_sector_signature: Option<Vec<isize>>,
    #[serde(rename = "threshold_CT_settings")]
    pub threshold_ct_settings: TriBoxTriCFFSectoredCTSettings,
}

pub struct TriBoxTriCFFSectoredIntegrand {
    pub settings: Settings,
    pub supergraph: SuperGraph,
    pub n_dim: usize,
    pub integrand_settings: TriBoxTriCFFSectoredSettings,
    pub event_manager: EventManager,
    pub sampling_rot: Option<[[isize; 3]; 3]>,
    pub hard_coded_rules: Vec<SectoringRule>,
}

#[derive(Debug, Clone)]
pub struct TriBoxTriCFFSectoredComputationCachePerCut<T: FloatLike> {
    pub onshell_edge_momenta: Vec<LorentzVector<T>>,
    pub rescaled_loop_momenta: Vec<LorentzVector<T>>,
    // 0 means observable regected that cut, 1 means observable passed on that cut and -1 means cff term does not contribut to that cut
    pub sector_signature: Vec<isize>,
    pub cut_sg_e_surf_id: usize,
    pub cut_sg_e_surf: GenericESurfaceCache<T>,
    pub evt: Event,
    pub e_surf_cts: Vec<Vec<ESurfaceCT<T, GenericESurfaceCache<T>>>>,
    pub e_surf_caches: Vec<GenericESurfaceCache<T>>,
}

impl<T: FloatLike> TriBoxTriCFFSectoredComputationCachePerCut<T> {
    pub fn default() -> TriBoxTriCFFSectoredComputationCachePerCut<T> {
        TriBoxTriCFFSectoredComputationCachePerCut {
            onshell_edge_momenta: vec![],
            rescaled_loop_momenta: vec![],
            sector_signature: vec![],
            cut_sg_e_surf_id: 0,
            cut_sg_e_surf: GenericESurfaceCache::default(),
            evt: Event::default(),
            e_surf_cts: vec![],
            e_surf_caches: vec![],
        }
    }
}

#[derive(Debug)]
pub struct TriBoxTriCFFSectoredComputationCache<T: FloatLike> {
    pub external_momenta: Vec<LorentzVector<T>>,
    pub cut_caches: Vec<TriBoxTriCFFSectoredComputationCachePerCut<T>>,
    pub sector_signature: Vec<isize>,
    pub sampling_xs: Vec<f64>,
}

impl<T: FloatLike> TriBoxTriCFFSectoredComputationCache<T> {
    pub fn default() -> TriBoxTriCFFSectoredComputationCache<T> {
        TriBoxTriCFFSectoredComputationCache {
            external_momenta: vec![],
            cut_caches: vec![],
            sector_signature: vec![],
            sampling_xs: vec![],
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
impl TriBoxTriCFFSectoredIntegrand {
    pub fn new(
        settings: Settings,
        integrand_settings: TriBoxTriCFFSectoredSettings,
    ) -> TriBoxTriCFFSectoredIntegrand {
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
        let hard_coded_rules: Vec<SectoringRule> = if let Some(hard_coded_rules_file) =
            integrand_settings
                .threshold_ct_settings
                .sectoring_settings
                .hard_coded_rules_file
                .as_ref()
        {
            let f = File::open(&hard_coded_rules_file).unwrap_or_else(|_e| {
                panic!(
                    "Could not open hardcoded sectoring rules file {}",
                    hard_coded_rules_file
                )
            });
            serde_yaml::from_reader(f)
                .unwrap_or_else(|_e| panic!("Could not parse hardcoded sectoring rules file"))
        } else {
            vec![]
        };
        TriBoxTriCFFSectoredIntegrand {
            settings,
            supergraph: sg,
            n_dim: n_dim,
            integrand_settings,
            event_manager,
            sampling_rot: None,
            hard_coded_rules,
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
        edge_ids: &Vec<usize>,
        cache: &TriBoxTriCFFSectoredComputationCache<T>,
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
        cache: &TriBoxTriCFFSectoredComputationCache<T>,
        loop_momenta: &Vec<LorentzVector<T>>,
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
            if e_surf_side == NOSIDE {
                // For the present case we can hardcode that the tree E-surface is always pinched.
                let mut tree_e_surf_cache = self.build_e_surface_for_edges(
                    &sg_lmb,
                    &e_surf.edge_ids,
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
            let mut e_surf_cache = self.build_e_surface_for_edges(
                &amplitude_for_sides[e_surf_side]
                    .lmb_edges
                    .iter()
                    .map(|e| e.id)
                    .collect::<Vec<_>>(),
                &e_surf_loop_edges,
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
        cache: &TriBoxTriCFFSectoredComputationCache<T>,
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

    fn fill_cache<T: FloatLike>(
        &mut self,
        loop_momenta: &Vec<LorentzVector<T>>,
        cache: &mut TriBoxTriCFFSectoredComputationCache<T>,
    ) {
        if self.settings.general.debug > 3 {
            println!(
                "{}",
                format!("  > Now starting global analysis of thresholds over all cuts").bold()
            );
        }

        // let mut sorted_cut_ids = (0..self.supergraph.cuts.len()).collect::<Vec<_>>();
        // sorted_cut_ids.sort_unstable_by(|i, j| {
        //     (self.supergraph.cuts[*i].left_amplitude.n_loop
        //         + self.supergraph.cuts[*i].right_amplitude.n_loop)
        //         .partial_cmp(
        //             &(self.supergraph.cuts[*j].left_amplitude.n_loop
        //                 + self.supergraph.cuts[*j].right_amplitude.n_loop),
        //         )
        //         .unwrap()
        // });
        cache.cut_caches =
            vec![TriBoxTriCFFSectoredComputationCachePerCut::default(); self.supergraph.cuts.len()];
        // for i_cut in sorted_cut_ids {
        for i_cut in 0..self.supergraph.cuts.len() {
            let cut = &self.supergraph.cuts[i_cut];
            if self.settings.general.debug > 2 {
                println!(
                    "{}",
                    format!(
                        "  | Starting analysis of of cut #{}({})",
                        i_cut,
                        format!(
                            "{}",
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
                    )
                    .blue()
                    .bold()
                );
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
            if self.settings.general.debug > 4 {
                println!(
                    "{}",
                    format!(
                        "  > Found that cut #{} corresponds to E-surface with id {}",
                        i_cut, sg_cut_e_surf
                    )
                    .blue()
                );
            }
            cache.cut_caches[i_cut].cut_sg_e_surf_id = sg_cut_e_surf;

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
                &self.supergraph.cuts[i_cut]
                    .cut_edge_ids_and_flip
                    .iter()
                    .map(|(e_id, _flip)| *e_id)
                    .collect::<Vec<_>>(),
                &cache,
                // Nothing will be used from the loop momenta in this context because we specify all edges to be in the e surf basis here.
                // But this construction is useful when building the amplitude e-surfaces using the same function.
                loop_momenta,
                NOSIDE,
                &vec![-1, 0],
            );

            e_surface_cc_cut.t_scaling = e_surface_cc_cut.compute_t_scaling(&loop_momenta);
            cache.cut_caches[i_cut].cut_sg_e_surf = e_surface_cc_cut.clone();

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
            cache.cut_caches[i_cut].onshell_edge_momenta =
                onshell_edge_momenta_for_this_cut.clone();
            cache.cut_caches[i_cut].rescaled_loop_momenta = rescaled_loop_momenta.clone();
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
            cache.cut_caches[i_cut].evt = evt.clone();
            cache.cut_caches[i_cut].sector_signature = vec![-1; self.supergraph.cuts.len()];
            cache.cut_caches[i_cut].sector_signature[i_cut] =
                if self.event_manager.pass_selection(&mut evt) {
                    CUT_ACTIVE
                } else {
                    CUT_INACTIVE
                };
            if self.settings.general.debug > 4 {
                println!("      > Cut {}: Event for sector signature of same cut {}({}), which {}, is: {:?}",
                        format!("#{}",i_cut).blue(),
                        format!("#{}",i_cut).blue(),
                        format!(
                            "{}",
                            self.supergraph.cuts[i_cut].cut_edge_ids_and_flip
                                .iter()
                                .map(|(id, flip)| if *flip > 0 {
                                    format!("+{}", id)
                                } else {
                                    format!("-{}", id)
                                })
                                .collect::<Vec<_>>()
                                .join("|")
                        ).blue(),
                        if cache.cut_caches[i_cut].sector_signature[i_cut] == 1 {
                            format!("{}","PASSED").green()
                        } else {
                            format!("{}","FAILED").red()
                        },
                        evt
                    );
            }
            if cache.cut_caches[i_cut].sector_signature[i_cut] == 0 {
                if self.settings.general.debug > 2 {
                    println!(
                        "{}",
                        format!(
                            "  | Cut {}: Aborting analysis of thresholds of this cut as it does not pass its own cut selection.",
                            i_cut,
                        )
                        .red()
                        .bold()
                    );
                }
                continue;
            }
            for j_cut in 0..self.supergraph.cuts.len() {
                if j_cut == i_cut {
                    continue;
                }
                let mut j_cut_evt = self.event_manager.create_event(
                    vec![cache.external_momenta[0]],
                    self.supergraph.cuts[j_cut]
                        .cut_edge_ids_and_flip
                        .iter()
                        .map(|(i_edge, flip)| {
                            let mut edge_mom = onshell_edge_momenta_for_this_cut[*i_edge]
                                * Into::<T>::into(*flip as f64);
                            edge_mom.t = edge_mom.t.abs();
                            edge_mom
                        })
                        .collect::<Vec<_>>(),
                );
                cache.cut_caches[i_cut].sector_signature[j_cut] =
                    if self.event_manager.pass_selection(&mut j_cut_evt) {
                        CUT_ACTIVE
                    } else {
                        CUT_INACTIVE
                    };
                if self.settings.general.debug > 4 {
                    println!("      > Cut {}: Event for sector signature of other cut {}({}), which {}, is: {:?}",
                        format!("#{}",i_cut).blue(),
                        format!("#{}",j_cut).blue(),
                        format!(
                            "{}",
                            self.supergraph.cuts[j_cut].cut_edge_ids_and_flip
                                .iter()
                                .map(|(id, flip)| if *flip > 0 {
                                    format!("+{}", id)
                                } else {
                                    format!("-{}", id)
                                })
                                .collect::<Vec<_>>()
                                .join("|")
                        ).blue(),
                        if cache.cut_caches[i_cut].sector_signature[j_cut] == CUT_ACTIVE {
                            format!("{}","PASSED").green()
                        } else {
                            format!("{}","FAILED").red()
                        },
                        j_cut_evt
                    );
                }
            }
        }

        for i_cut in 0..self.supergraph.cuts.len() {
            if self
                .integrand_settings
                .threshold_ct_settings
                .apply_original_event_selection_to_cts
            {
                if cache.cut_caches[i_cut].sector_signature[i_cut] == CUT_INACTIVE {
                    if self.settings.general.debug > 2 {
                        println!(
                            "{}",
                            format!(
                                "      > Cut {}: Aborting analysis of this cut as it does not pass its own cut selection.",
                                i_cut,
                            )
                            .red()
                            .bold()
                        );
                    }
                    continue;
                }
            }
            let cut = &self.supergraph.cuts[i_cut];
            if self.settings.general.debug > 2 {
                println!(
                    "{}",
                    format!(
                        "  | Now analyzing thresholds of cut #{}({}) with sector signature {:?}",
                        i_cut,
                        format!(
                            "{}",
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
                        cache.cut_caches[i_cut].sector_signature
                    )
                    .blue()
                    .bold()
                );
            }
            // println!(
            //     "evt for this cut #{}: {:?}",
            //     i_cut, cache.cut_caches[i_cut].evt
            // );

            cache.cut_caches[i_cut].e_surf_caches = self.build_e_surfaces(
                &cache.cut_caches[i_cut].onshell_edge_momenta,
                &cache,
                &cache.cut_caches[i_cut].rescaled_loop_momenta,
                i_cut,
            );
            if self.settings.general.debug > 3 {
                let e_surf_caches = &cache.cut_caches[i_cut].e_surf_caches;
                if e_surf_caches.len() == 0 {
                    println!(
                        "    | All e_surf caches:      {}",
                        format!("{}", "None").green()
                    );
                } else {
                    let mut n_e_surf_to_subtract = 0;
                    for es in e_surf_caches {
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
                            .filter(|(i_esurf, es)| self.settings.general.debug > 4
                                || es.exists && !es.pinched)
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

            let mut e_surf_cts = vec![];
            for (e_surf_id, e_surf) in cache.cut_caches[i_cut].e_surf_caches.iter().enumerate() {
                if !self.integrand_settings.threshold_ct_settings.enabled
                    || e_surf_id == cache.cut_caches[i_cut].cut_sg_e_surf_id
                    || e_surf.side == NOSIDE
                    || !e_surf.exists
                    || e_surf.pinched
                {
                    e_surf_cts.push(vec![]);
                    continue;
                }
                if self.settings.general.debug > 3 {
                    println!(
                        "    | Now subtracting E-surface {}",
                        format!(
                            "{} : {:?}",
                            e_surf_str(
                                e_surf_id,
                                &self.supergraph.supergraph_cff.cff_expression.e_surfaces
                                    [e_surf_id]
                            )
                            .bold()
                            .red(),
                            cache.cut_caches[i_cut].e_surf_caches[e_surf_id]
                        )
                    );
                }
                e_surf_cts
                    .push(self.prepare_threshold_cts_for_e_surf_and_cut(e_surf_id, i_cut, &cache));
            }
            cache.cut_caches[i_cut].e_surf_cts = e_surf_cts;
        }
        cache.sector_signature = cache
            .cut_caches
            .iter()
            .enumerate()
            .map(|(i_cut, cut_cache)| cut_cache.sector_signature[i_cut])
            .collect::<Vec<_>>();
        if self.settings.general.debug > 3 {
            println!("{}",
                format!("  > Finished global analysis of thresholds over all cuts resulting in sector signature: {:?}",cache.sector_signature).bold()
            );
        }
    }

    fn prepare_threshold_cts_for_e_surf_and_cut<T: FloatLike>(
        &mut self,
        e_surf_id: usize,
        i_cut: usize,
        cache: &TriBoxTriCFFSectoredComputationCache<T>,
    ) -> Vec<ESurfaceCT<T, GenericESurfaceCache<T>>> {
        let mut all_new_cts = vec![];

        let cut = &self.supergraph.cuts[i_cut];

        let sectoring_settings = &self
            .integrand_settings
            .threshold_ct_settings
            .sectoring_settings;

        let e_surf_cache = &cache.cut_caches[i_cut].e_surf_caches;
        let e_surf = &e_surf_cache[e_surf_id];

        // Quite some gynmastic needs to take place in order to dynamqically build the right basis for solving this E-surface CT
        // I semi-hard-code it for now
        // In general this is more complicated and would involve an actual change of basis, but here we can do it like this
        let max_loop_indices_for_this_ct = e_surf.get_e_surface_basis_indices().clone();

        let side = e_surf.side;
        let other_side = if side == LEFT { RIGHT } else { LEFT };
        let amplitudes_pair = [&cut.left_amplitude, &cut.right_amplitude];
        let max_other_side_loop_indices = self.loop_indices_in_edge_ids(
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

        let max_edges_to_consider_when_building_subtracted_e_surf =
            self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id]
                .edge_ids
                .iter()
                .filter(|&e_id| amplitudes_pair[side].edges.iter().any(|e| e.id == *e_id))
                .map(|e_id| *e_id)
                .collect::<Vec<_>>();
        let mut all_edges_and_loop_indices_to_solve_ct_in = vec![(
            max_edges_to_consider_when_building_subtracted_e_surf.clone(),
            max_loop_indices_for_this_ct.clone(),
        )];

        let mut all_other_side_loop_indices_to_solve_ct_in =
            vec![max_other_side_loop_indices.clone()];

        if max_loop_indices_for_this_ct.len() > 1 {
            for (i_other_cut, cut) in self.supergraph.cuts.iter().enumerate() {
                let this_cut_amplitudes_pair = [&cut.left_amplitude, &cut.right_amplitude];

                let sg_level_e_surf_intersection_n_edges =
                    self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id]
                        .edge_ids
                        .len()
                        + cut.cut_edge_ids_and_flip.len()
                        - 2 * cut
                            .cut_edge_ids_and_flip
                            .iter()
                            .filter(|(e_id, _flip)| {
                                self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id]
                                    .edge_ids
                                    .contains(e_id)
                            })
                            .count();

                // Hack to ignore anti-selection vs cut for whom this e_surf ID would be a pinched surface
                if sg_level_e_surf_intersection_n_edges <= 3 {
                    continue;
                }

                let new_entry = (
                    max_edges_to_consider_when_building_subtracted_e_surf
                        .iter()
                        .filter(|&e_id| {
                            this_cut_amplitudes_pair
                                .iter()
                                .any(|amp| amp.edges.iter().any(|e| e.id == *e_id))
                        })
                        .map(|e_id| *e_id)
                        .collect::<Vec<_>>(),
                    max_loop_indices_for_this_ct
                        .iter()
                        .filter(|&li| {
                            this_cut_amplitudes_pair.iter().any(|amp| {
                                amp.lmb_edges
                                    .iter()
                                    .any(|e| self.supergraph.edges[e.id].signature.0[*li] != 0)
                            })
                        })
                        .map(|li| *li)
                        .collect::<Vec<_>>(),
                );
                // println!(
                //     "Considering cut #{}({}) generating entry: {:?}",
                //     i_cut,
                //     format!(
                //         "{}",
                //         self.supergraph.cuts[i_cut]
                //             .cut_edge_ids_and_flip
                //             .iter()
                //             .map(|(id, flip)| if *flip > 0 {
                //                 format!("+{}", id)
                //             } else {
                //                 format!("-{}", id)
                //             })
                //             .collect::<Vec<_>>()
                //             .join("|")
                //     )
                //     .blue(),
                //     new_entry
                // );
                if new_entry.1.len() == 0
                    || new_entry.0.len() < 2
                    || all_edges_and_loop_indices_to_solve_ct_in.contains(&new_entry)
                {
                    continue;
                }
                all_edges_and_loop_indices_to_solve_ct_in.push(new_entry);
            }
        }

        if max_other_side_loop_indices.len() > 1 {
            panic!("More than one other side loop indices to solve CT in. This is not implemented yet, but the idea is simple and just amounts to consider the powerset.");
        } else if max_other_side_loop_indices.len() == 1 {
            all_other_side_loop_indices_to_solve_ct_in.push(vec![]);
        }

        let scaled_loop_momenta_in_sampling_basis = &cache.cut_caches[i_cut].rescaled_loop_momenta;
        let onshell_edge_momenta_for_this_cut = &cache.cut_caches[i_cut].onshell_edge_momenta;

        if self.settings.general.debug > 3 {
            println!("    | Considering the following list of combinations of  edges and loop indices to solve CT in: {:?}", all_edges_and_loop_indices_to_solve_ct_in);
            println!(
                "    | and the following list of other-side loop indices to solve CT in: {:?}",
                all_other_side_loop_indices_to_solve_ct_in
            );
        }
        for (edges_to_consider_when_building_subtracted_e_surf, loop_indices_for_this_ct) in
            all_edges_and_loop_indices_to_solve_ct_in.iter()
        {
            for other_side_loop_indices in all_other_side_loop_indices_to_solve_ct_in.iter() {
                if !self
                    .integrand_settings
                    .threshold_ct_settings
                    .include_cts_solved_in_two_loop_space
                    && loop_indices_for_this_ct.len() > 1
                {
                    // println!(
                    //     "Consdering edges_to_consider_when_building_subtracted_e_surf={:?}, loop_indices_for_this_ct={:?}, other_side_loop_indices={:?}",
                    //     edges_to_consider_when_building_subtracted_e_surf, loop_indices_for_this_ct,other_side_loop_indices
                    // );
                    if self.settings.general.debug > 3 {
                        println!(
                            "      > {} two-loop CT in same side loop indices {} and other side loop indices {} for e-surface {}",
                            format!("{}","User disabled").red(),
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
                    continue;
                }

                if !self
                    .integrand_settings
                    .threshold_ct_settings
                    .include_cts_solved_in_one_loop_subspace
                    && loop_indices_for_this_ct.len() < 2
                    && max_loop_indices_for_this_ct.len() > 1
                {
                    if self.settings.general.debug > 3 {
                        println!(
                            "      > {} one-loop subspace projected CT in same side loop indices {} and other side loop indices {} for e-surface {}",
                            format!("{}","User disabled").red(),
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
                    continue;
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

                let mut loop_momenta_in_e_surf_basis =
                    scaled_loop_momenta_in_sampling_basis.clone();
                for loop_index in loop_indices_for_this_ct.iter() {
                    loop_momenta_in_e_surf_basis[*loop_index] -= center_shifts[*loop_index];
                }

                // The building of the E-surface should be done more generically and efficiently, but here in this simple case we can do it this way
                let mut subtracted_e_surface = if edges_to_consider_when_building_subtracted_e_surf
                    == &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id].edge_ids
                {
                    e_surf_cache[e_surf_id].clone()
                } else {
                    let mut e_surf_to_build = self.build_e_surface_for_edges(
                        &loop_indices_for_this_ct
                            .iter()
                            .map(|li| self.supergraph.supergraph_cff.lmb_edges[*li].id)
                            .collect::<Vec<_>>(),
                        &edges_to_consider_when_building_subtracted_e_surf,
                        &cache,
                        // Nothing will be used from the loop momenta in this context because we specify all edges to be in the e surf basis here.
                        // But this construction is useful when building the amplitude e-surfaces using the same function.
                        scaled_loop_momenta_in_sampling_basis,
                        side,
                        &vec![-1, 0],
                    );
                    for i_e in &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id]
                        .edge_ids
                    {
                        if !edges_to_consider_when_building_subtracted_e_surf.contains(i_e) {
                            e_surf_to_build.e_shift +=
                                onshell_edge_momenta_for_this_cut[*i_e].t.abs();
                        }
                    }
                    (e_surf_to_build.exists, e_surf_to_build.pinched) =
                        e_surf_to_build.does_exist();
                    if self.settings.general.debug > 3 {
                        e_surf_to_build.eval =
                            e_surf_to_build.eval(&scaled_loop_momenta_in_sampling_basis);
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
                    continue;
                }

                let center_eval = subtracted_e_surface.eval(&center_shifts);
                assert!(center_eval < T::zero());

                // Change the parametric equation of the subtracted E-surface to the CT basis
                subtracted_e_surface.adjust_loop_momenta_shifts(&center_coordinates);

                subtracted_e_surface.t_scaling =
                    subtracted_e_surface.compute_t_scaling(&loop_momenta_in_e_surf_basis);
                const _THRESHOLD: f64 = 1.0e-8;
                if subtracted_e_surface.t_scaling[MINUS] > T::zero() {
                    if subtracted_e_surface.t_scaling[MINUS]
                        > Into::<T>::into(_THRESHOLD * self.settings.kinematics.e_cm as f64)
                    {
                        panic!(
                            "Unexpected positive t-scaling for negative solution: {:+.e} arising from sample {:?} and for e_surf:\n{:?}",
                            subtracted_e_surface.t_scaling[MINUS], cache.sampling_xs, subtracted_e_surface
                        );
                    }
                    if self.settings.general.debug > 0 {
                        println!("{}",format!(
                            "WARNING:: Unexpected positive t-scaling for negative solution: {:+.e} arising from sample {:?} and for e_surf:\n{:?}",
                            subtracted_e_surface.t_scaling[MINUS], cache.sampling_xs,subtracted_e_surface
                        ).bold().red());
                    }
                    subtracted_e_surface.t_scaling[MINUS] =
                        -Into::<T>::into(_THRESHOLD * self.settings.kinematics.e_cm as f64);
                }
                if subtracted_e_surface.t_scaling[PLUS] < T::zero() {
                    if subtracted_e_surface.t_scaling[PLUS]
                        < -Into::<T>::into(_THRESHOLD * self.settings.kinematics.e_cm as f64)
                    {
                        panic!(
                            "Unexpected negative t-scaling for positive solution: {:+.e} arising from sample {:?} and for e_surf:\n{:?}",
                            subtracted_e_surface.t_scaling[PLUS], cache.sampling_xs, subtracted_e_surface
                        );
                    }
                    if self.settings.general.debug > 0 {
                        println!("{}",format!(
                            "WARNING: Unexpected negative t-scaling for positive solution: {:+.e} arising from sample {:?} and for e_surf:\n{:?}",
                            subtracted_e_surface.t_scaling[PLUS], cache.sampling_xs,subtracted_e_surface).bold().red()
                        );
                    }
                    subtracted_e_surface.t_scaling[PLUS] =
                        Into::<T>::into(_THRESHOLD * self.settings.kinematics.e_cm as f64);
                }
                assert!(subtracted_e_surface.t_scaling[MINUS] <= T::zero());
                assert!(subtracted_e_surface.t_scaling[PLUS] >= T::zero());

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
                let solutions_to_consider =
                    match self.integrand_settings.threshold_ct_settings.variable {
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

                for ct_level_ref in &ct_levels_to_consider {
                    let ct_level = *ct_level_ref;
                    if other_side_loop_indices.len() > 0 && ct_level == AMPLITUDE_LEVEL_CT {
                        continue;
                    }
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
                                "      > {} {} CT ({} solution, t={:+.16e}) in same side loop indices {} and other side loop indices {} for e-surface {}",
                                format!("{}","Solving").green(),
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
                        let mut loop_momenta_star_in_e_surf_basis =
                            loop_momenta_in_e_surf_basis.clone();
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
                            loop_momenta_star_in_sampling_basis[*loop_index] +=
                                center_shifts[*loop_index];
                        }
                        if ct_level == SUPERGRAPH_LEVEL_CT {
                            for loop_index in other_side_loop_indices.iter() {
                                loop_momenta_star_in_sampling_basis[*loop_index] +=
                                    center_shifts[*loop_index];
                            }
                        }

                        let mut loop_momenta_star_in_sampling_basis_for_each_cut = vec![];
                        for (_other_i_cut, cut) in self.supergraph.cuts.iter().enumerate() {
                            let mut loop_momenta_star_in_sampling_basis_for_this_cut =
                                loop_momenta_star_in_sampling_basis.clone();

                            let mut e_surface_for_this_cc_cut = self.build_e_surface_for_edges(
                                &self
                                    .supergraph
                                    .supergraph_cff
                                    .lmb_edges
                                    .iter()
                                    .map(|e| e.id)
                                    .collect::<Vec<_>>(),
                                &cut.cut_edge_ids_and_flip
                                    .iter()
                                    .map(|(e_id, _flip)| *e_id)
                                    .collect::<Vec<_>>(),
                                &cache,
                                // Nothing will be used from the loop momenta in this context because we specify all edges to be in the e surf basis here.
                                // But this construction is useful when building the amplitude e-surfaces using the same function.
                                &loop_momenta_star_in_sampling_basis,
                                NOSIDE,
                                &vec![-1, 0],
                            );

                            e_surface_for_this_cc_cut.t_scaling = e_surface_for_this_cc_cut
                                .compute_t_scaling(&loop_momenta_star_in_sampling_basis);
                            for lm in loop_momenta_star_in_sampling_basis_for_this_cut.iter_mut() {
                                lm.x *= e_surface_for_this_cc_cut.t_scaling[0];
                                lm.y *= e_surface_for_this_cc_cut.t_scaling[0];
                                lm.z *= e_surface_for_this_cc_cut.t_scaling[0];
                            }
                            loop_momenta_star_in_sampling_basis_for_each_cut
                                .push(loop_momenta_star_in_sampling_basis_for_this_cut);
                        }

                        // println!(
                        //     "scaled_loop_momenta_in_sampling_basis={:?}",
                        //     scaled_loop_momenta_in_sampling_basis
                        // );
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
                            if self.settings.general.debug > 4 {
                                println!("{}",
                                    format!("{}","      > Supergraph-level counterterm didn't pass the proximity threshold, skipping it.").red()
                                );
                            }
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
                                if ((t - t_star) / t_star) * ((t - t_star) / t_star)
                                    > sliver * sliver
                                {
                                    include_integrated_ct = false;
                                }
                            }
                        }
                        if !include_local_ct && !include_integrated_ct {
                            if self.settings.general.debug > 4 {
                                println!("{}",
                                    format!("{}","      > Amplitude-level Ccunterterm didn't pass the proximity threshold, skipping it.").red()
                                );
                            }
                            continue;
                        }

                        let onshell_edge_momenta_for_this_ct = self.evaluate_onshell_edge_momenta(
                            &loop_momenta_star_in_sampling_basis,
                            &cache.external_momenta,
                            &self
                                .supergraph
                                .supergraph_cff
                                .edges
                                .iter()
                                .map(|e| (e.id, 1_isize))
                                .collect::<Vec<_>>(),
                        );

                        let onshell_edge_momenta_for_this_ct_for_each_cut =
                            loop_momenta_star_in_sampling_basis_for_each_cut
                                .iter()
                                .map(|lms| {
                                    self.evaluate_onshell_edge_momenta(
                                        &lms,
                                        &cache.external_momenta,
                                        &self
                                            .supergraph
                                            .supergraph_cff
                                            .edges
                                            .iter()
                                            .map(|e| (e.id, 1_isize))
                                            .collect::<Vec<_>>(),
                                    )
                                })
                                .collect::<Vec<_>>();

                        // println!(
                        //     "scaled_loop_momenta_in_sampling_basis={:?}",
                        //     scaled_loop_momenta_in_sampling_basis
                        // );
                        // println!(
                        //     "onshell_edge_momenta_for_this_ct={:?}",
                        //     onshell_edge_momenta_for_this_ct
                        // );

                        let ct_i_cut =
                        match self.supergraph.cuts.iter().enumerate().find(|(_i, c)| {
                            c.cut_edge_ids_and_flip.iter().all(|(e_id, _flip)| {
                                self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id]
                                    .edge_ids
                                    .contains(e_id)
                            })
                        }) {
                            Some((i_c, c)) => i_c,
                            None => panic!(
                            "Could not find the Cutkosky cut corresponding to this subtracted E-surface:\n{}",
                            format!(
                                "{}",
                                e_surf_str(
                                    e_surf_id,
                                    &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],
                                )
                            )
                        ),
                        };
                        let mut ct_sector_signature = vec![CUT_ABSENT; self.supergraph.cuts.len()];
                        // We can immediately set the sector signature for this CT to the one of the cut it stems from
                        ct_sector_signature[i_cut] =
                            cache.cut_caches[i_cut].sector_signature[i_cut];
                        // We should not immediately remove this CT if one of the sector signature element does not match
                        // that of the event because it may be that the cFF term consider does not contribute to that particular
                        // cut, so that we should not consider it. This is however obviously never the case for the cut corresponding to
                        // this CT itself and the cutkosky cut it stems from, so we can apply those immediately
                        // However, when debugging it's best to leave it here and let it be removed later for each cFF term
                        let mut cuts_for_signature = if self.settings.general.debug == 0 {
                            vec![(ct_i_cut, true)]
                        } else {
                            vec![(ct_i_cut, false)]
                        };
                        for sig_i_cut in 0..self.supergraph.cuts.len() {
                            if sig_i_cut != i_cut && sig_i_cut != ct_i_cut {
                                cuts_for_signature.push((sig_i_cut, false));
                            }
                        }
                        let mut veto_ct = false;
                        for (sig_i_cut, do_anti_select) in cuts_for_signature {
                            let mut ct_evt_for_sig_i_cut = self.event_manager.create_event(
                                vec![cache.external_momenta[0]],
                                self.supergraph.cuts[sig_i_cut]
                                    .cut_edge_ids_and_flip
                                    .iter()
                                    .map(|(i_edge, flip)| {
                                        let mut edge_mom =
                                            onshell_edge_momenta_for_this_ct_for_each_cut
                                                [sig_i_cut][*i_edge]
                                                * Into::<T>::into(*flip as f64);
                                        edge_mom.t = edge_mom.t.abs();
                                        edge_mom
                                    })
                                    .collect::<Vec<_>>(),
                            );
                            ct_sector_signature[sig_i_cut] =
                                if self.event_manager.pass_selection(&mut ct_evt_for_sig_i_cut) {
                                    CUT_ACTIVE
                                } else {
                                    CUT_INACTIVE
                                };
                            if self.settings.general.debug > 4 {
                                println!("      > E-surface {}: Event for sector signature of cut {}({}), which {} ({}), is: {:?}",
                                    format!("#{}",e_surf_id).blue(),
                                    format!("#{}",sig_i_cut).blue(),
                                    format!(
                                        "{}",
                                        self.supergraph.cuts[sig_i_cut].cut_edge_ids_and_flip
                                            .iter()
                                            .map(|(id, flip)| if *flip > 0 {
                                                format!("+{}", id)
                                            } else {
                                                format!("-{}", id)
                                            })
                                            .collect::<Vec<_>>()
                                            .join("|")
                                    ).blue(),
                                    if ct_sector_signature[sig_i_cut] == CUT_ACTIVE {
                                        format!("{}","PASSED").green()
                                    } else {
                                        format!("{}","FAILED").red()
                                    },
                                    if do_anti_select {
                                        format!("{}","vetoing test").bold()
                                    } else {
                                        format!("{}","non-vetoing test").bold()
                                    },
                                    ct_evt_for_sig_i_cut
                                );
                            }
                            if self
                                .integrand_settings
                                .threshold_ct_settings
                                .sectoring_settings
                                .anti_select_threshold_against_observable
                                && do_anti_select
                                && ct_sector_signature[sig_i_cut] == CUT_ACTIVE
                            {
                                veto_ct = true;
                                // println!(
                                //     "evt for sel of cut #{}: {:?}",
                                //     sig_i_cut, ct_evt_for_sig_i_cut
                                // );
                                if self.settings.general.debug > 3 {
                                    println!(
                                        "      > {} {} CT ({} solution, t={:+.16e}) in same side loop indices {} and other side loop indices {} for e-surface {} because it failed selection on cut {}({})",
                                        format!("{}","Removing").red(),
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
                                        e_surf_str(e_surf_id, &self.supergraph.supergraph_cff.cff_expression.e_surfaces[e_surf_id],).red(),
                                        format!("#{}",sig_i_cut).red(),
                                        format!(
                                            "{}",
                                            self.supergraph.cuts[sig_i_cut].cut_edge_ids_and_flip
                                                .iter()
                                                .map(|(id, flip)| if *flip > 0 {
                                                    format!("+{}", id)
                                                } else {
                                                    format!("-{}", id)
                                                })
                                                .collect::<Vec<_>>()
                                                .join("|")
                                        ).red(),
                                    );
                                }
                                break;
                            }
                        }
                        if veto_ct {
                            continue;
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

                        // Construct e-surfaces relevant to each CT.
                        let mut loop_indices_projected_out = max_loop_indices_for_this_ct
                            .iter()
                            .filter(|li| !loop_indices_for_this_ct.contains(li))
                            .map(|li| *li)
                            .collect::<Vec<_>>();
                        loop_indices_projected_out.extend(
                            max_other_side_loop_indices
                                .iter()
                                .filter(|li| !other_side_loop_indices.contains(li)),
                        );
                        let mut pinched_e_surf_ids_active_in_projected_out_subspace = vec![];
                        let mut pinched_e_surf_ids_active_solved_subspace = vec![];
                        let mut non_pinched_e_surf_ids_active_in_projected_out_subspace = vec![];
                        let mut non_pinched_e_surf_ids_active_solved_subspace = vec![];
                        for (j_surf, a_e_surf) in e_surface_caches_for_this_ct[0].iter().enumerate()
                        {
                            let a_e_surf_loop_indices = a_e_surf.get_loop_indices_dependence();
                            if j_surf == e_surf_id
                                || a_e_surf.side == NOSIDE
                                || (!a_e_surf.exists && !a_e_surf.pinched)
                                || a_e_surf_loop_indices.iter().all(|li| {
                                    !max_loop_indices_for_this_ct.contains(li)
                                        && !max_other_side_loop_indices.contains(li)
                                })
                            {
                                continue;
                            }
                            let do_projected_out_indices_overlap = a_e_surf_loop_indices
                                .iter()
                                .any(|li| loop_indices_projected_out.contains(li));
                            let do_active_indices_overlap = loop_indices_for_this_ct
                                .iter()
                                .any(|li| a_e_surf_loop_indices.contains(li))
                                || other_side_loop_indices
                                    .iter()
                                    .any(|li| a_e_surf_loop_indices.contains(li));

                            if do_active_indices_overlap {
                                if a_e_surf.pinched {
                                    pinched_e_surf_ids_active_solved_subspace.push(j_surf);
                                } else {
                                    non_pinched_e_surf_ids_active_solved_subspace.push(j_surf);
                                }
                            } else if do_projected_out_indices_overlap {
                                if !a_e_surf.pinched {
                                    non_pinched_e_surf_ids_active_in_projected_out_subspace
                                        .push(j_surf);
                                } else {
                                    pinched_e_surf_ids_active_in_projected_out_subspace
                                        .push(j_surf);
                                }
                            }
                        }

                        // The factor (t - t_star) will be included at the end because we need the same quantity for the integrated CT
                        // let e_surf_derivative = e_surf_cache[side][e_surf_id]
                        //     .norm(&vec![
                        //         loop_momenta_star_in_sampling_basis[loop_index_for_this_ct],
                        //     ])
                        //     .spatial_dot(&loop_momenta_star_in_e_surf_basis[loop_index_for_this_ct])

                        let e_surf_derivative = subtracted_e_surface
                            .t_der(&loop_momenta_star_in_sampling_basis)
                            / r_star;
                        // Identifying the residue in t, with r=e^t means that we must drop the r_star normalisation in the expansion.
                        let e_surf_expanded =
                            match self.integrand_settings.threshold_ct_settings.variable {
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
                        if self.settings.general.debug > 4 && h_function_wgt == T::zero() {
                            println!("{}",
                                format!("{}","      > H-function weight for this CT is exactly zero, likely because it is very far from being on threshold.").red()
                            );
                        }
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
                            if self.settings.general.debug > 4 {
                                println!("{}",
                                    format!("{}","      > Amplitude-level local counterterm didn't pass the proximity threshold, setting its weight to zero.").red()
                                );
                            }
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
                            loop_momenta_star: loop_momenta_star_in_sampling_basis, // Not used at the moment, could be dropped
                            onshell_edges: onshell_edge_momenta_for_this_ct,
                            e_surface_evals: e_surface_caches_for_this_ct,
                            solution_type,        // Only for monitoring purposes
                            cff_evaluations,      // Dummy in this case
                            cff_pinch_dampenings, // Dummy in this case
                            integrated_ct,
                            ct_level,
                            loop_indices_solved,
                            ct_sector_signature,
                            e_surface_analysis: ESurfaceCTAnalysis {
                                pinched_e_surf_ids_active_in_projected_out_subspace,
                                pinched_e_surf_ids_active_solved_subspace,
                                non_pinched_e_surf_ids_active_in_projected_out_subspace,
                                non_pinched_e_surf_ids_active_solved_subspace,
                            },
                        });
                    }
                }
            }
        }

        if self.settings.general.debug > 5 {
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

    fn evaluate_cut<T: FloatLike>(
        &mut self,
        i_cut: usize,
        loop_momenta: &Vec<LorentzVector<T>>,
        overall_sampling_jac: T,
        cache: &TriBoxTriCFFSectoredComputationCache<T>,
        selected_sg_cff_term: Option<usize>,
    ) -> (
        Complex<T>,
        Complex<T>,
        Vec<Complex<T>>,
        Vec<Complex<T>>,
        Option<ClosestESurfaceMonitor<T>>,
        Option<ClosestESurfaceMonitor<T>>,
    ) {
        let sg_cut_e_surf = cache.cut_caches[i_cut].cut_sg_e_surf_id;

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

        let e_surface_cc_cut = &cache.cut_caches[i_cut].cut_sg_e_surf;
        let rescaled_loop_momenta = &cache.cut_caches[i_cut].rescaled_loop_momenta;

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
        let mut onshell_edge_momenta_for_this_cut =
            cache.cut_caches[i_cut].onshell_edge_momenta.clone();

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
        let original_event_did_pass_selection = self.event_manager.add_event(evt);
        if self
            .integrand_settings
            .threshold_ct_settings
            .apply_original_event_selection_to_cts
            && !original_event_did_pass_selection
        {
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

            let mut cut_sector_signature = cache.sector_signature.clone();
            let mut selected_sector_signature = if let Some(user_selected_sector_signature) =
                self.integrand_settings.selected_sector_signature.as_ref()
            {
                user_selected_sector_signature.clone()
            } else {
                cut_sector_signature.clone()
            };

            for i_sig_cut in 0..self.supergraph.cuts.len() {
                if !self.supergraph.supergraph_cff.cff_expression.terms[i_cff]
                    .contains_e_surf_id(cache.cut_caches[i_sig_cut].cut_sg_e_surf_id)
                {
                    cut_sector_signature[i_sig_cut] = CUT_ABSENT;
                    selected_sector_signature[i_sig_cut] = CUT_ABSENT;
                }
            }

            if self.integrand_settings.selected_sector_signature.is_some() {
                if cut_sector_signature != selected_sector_signature {
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
            let e_surf_caches = &cache.cut_caches[i_cut].e_surf_caches;

            if self.settings.general.debug > 0 {
                let cff_term = &self.supergraph.supergraph_cff.cff_expression.terms[i_cff];
                for (i_surf, e_surf_cache) in e_surf_caches.iter().enumerate() {
                    if i_surf == sg_cut_e_surf || !cff_term.contains_e_surf_id(i_surf) {
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
                if self.settings.general.debug > 3 {
                    println!(
                        "   > Now selecting active counterterms (evt sector signature for this cut: {})",
                        format!("{:?}", cut_sector_signature).green()
                    )
                }
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
                        if !self.supergraph.supergraph_cff.cff_expression.terms[i_cff]
                            .contains_e_surf_id(e_surf_id)
                        {
                            continue;
                        }
                        let new_cts = &cache.cut_caches[i_cut].e_surf_cts[e_surf_id];
                        for ct in new_cts {
                            let mut this_ct_sector_signature = ct.ct_sector_signature.clone();
                            for i_sig_cut in 0..self.supergraph.cuts.len() {
                                if !self.supergraph.supergraph_cff.cff_expression.terms[i_cff]
                                    .contains_e_surf_id(
                                        cache.cut_caches[i_sig_cut].cut_sg_e_surf_id,
                                    )
                                {
                                    this_ct_sector_signature[i_sig_cut] = CUT_ABSENT;
                                }
                            }

                            if self.settings.general.debug > 4 {
                                println!(
                                    "   > cFF Evaluation #{} : Testing activation of CT for {} E-surface {} solved in {} with sector signature {}",
                                    format!("{}", i_cff).green(),
                                    format!(
                                        "{}|{}|{}",
                                        if side == LEFT { "L" } else { "R" },
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
                                    format!("{:?}", this_ct_sector_signature).blue()
                                );
                            }

                            let loop_indices_solved = if side == LEFT {
                                &ct.loop_indices_solved.0
                            } else {
                                &ct.loop_indices_solved.1
                            };

                            if self
                                .integrand_settings
                                .threshold_ct_settings
                                .sectoring_settings
                                .apply_hard_coded_rules
                            {
                                let mut keep_this_one_ct = true;
                                let mut reason_for_this_ct = format!("{}", "");
                                let mut rule_found = None;
                                let mut mc_factor = T::one();

                                // First apply anti-observable if requested
                                for (sig_i_cut, cut_cache) in cache.cut_caches.iter().enumerate() {
                                    let cut_str = if self.settings.general.debug > 3 {
                                        format!(
                                            "{}({})",
                                            format!("#{}", sig_i_cut).blue(),
                                            format!(
                                                "{}",
                                                self.supergraph.cuts[sig_i_cut]
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
                                            .blue()
                                        )
                                    } else {
                                        format!("{}", "")
                                    };
                                    if cut_cache.cut_sg_e_surf_id == ct.e_surf_id
                                        && self
                                            .integrand_settings
                                            .threshold_ct_settings
                                            .sectoring_settings
                                            .anti_select_threshold_against_observable
                                    {
                                        if this_ct_sector_signature[sig_i_cut] == CUT_ACTIVE {
                                            keep_this_one_ct = false;
                                            if self.settings.general.debug > 3 {
                                                reason_for_this_ct = format!("it contains the following cut {} which is the threshold itself and selected by the observable, so anti-selection of threshold kicks in.",
                                                    cut_str
                                                );
                                            }
                                            break;
                                        }
                                    }
                                }
                                if keep_this_one_ct {
                                    // Find a hard-coded rule
                                    for (i_rule, hc_rule) in
                                        self.hard_coded_rules.iter().enumerate()
                                    {
                                        if hc_rule
                                            .sector_signature
                                            .iter()
                                            .zip(this_ct_sector_signature.iter())
                                            .all(|(&target_sig, &sig)| {
                                                sig == target_sig
                                                    || (target_sig == -2 && sig != CUT_ACTIVE)
                                            })
                                        {
                                            for (i_rule_cut, hc_rule_for_cut) in
                                                hc_rule.rules_for_cut.iter().enumerate()
                                            {
                                                if hc_rule_for_cut.cut_id == i_cut {
                                                    let i_rule_ct = match hc_rule_for_cut.rules_for_ct.iter().enumerate().find(|(_i_rule_ct, hc_rule_for_cut_and_ct)| {
                                                    hc_rule_for_cut_and_ct.surf_id_subtracted == ct.e_surf_id && loop_indices_solved.iter().all(|li| hc_rule_for_cut_and_ct.loop_indices_this_ct_is_solved_in.contains(li)) && hc_rule_for_cut_and_ct.loop_indices_this_ct_is_solved_in.iter().all(|li| loop_indices_solved.contains(li))
                                                }) {
                                                    Some((i_rule_ct, _hc_rule_for_cut_and_ct)) => i_rule_ct,
                                                    _ => panic!("   | Could not find hard-coded rule for this CT for E-surface #{} solved in loop indices {:?} and whose signature ({:?}) and cut id ({}) are however specified.",
                                                        ct.e_surf_id,
                                                        loop_indices_solved,
                                                        this_ct_sector_signature,
                                                        i_cut
                                                    )
                                                };
                                                    if self.settings.general.debug > 3 {
                                                        println!(
                                                        "   | cFF Evaluation #{} : CT for {} E-surface {} solved in {} with sector signature {} will be handled by hard-coded rule #{}",
                                                        format!("{}", i_cff).green(),
                                                        format!(
                                                            "{}|{}|{}",
                                                            if side == LEFT { "L" } else { "R" },
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
                                                        format!("{:?}", this_ct_sector_signature).blue(),
                                                        format!("{}", i_rule).red()
                                                    );
                                                    }
                                                    if rule_found.is_some() {
                                                        panic!("Multiple hard-coded rules found for a counterterm!");
                                                    } else {
                                                        rule_found =
                                                            Some((i_rule, i_rule_cut, i_rule_ct));
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if let Some((i_rule, i_rule_cut, i_rule_ct)) = rule_found {
                                        let hc_rule = &self.hard_coded_rules[i_rule].rules_for_cut
                                            [i_rule_cut]
                                            .rules_for_ct[i_rule_ct];
                                        if self.settings.general.debug > 3 {
                                            println!(
                                            "     | Found the hard-coded rule with coordinates ({},{},{}) applying to this counterterm: {:?})",
                                            i_rule, i_rule_cut, i_rule_ct, hc_rule
                                        )
                                        }

                                        if hc_rule.enabled {
                                            mc_factor = if hc_rule
                                                .mc_factor
                                                .e_surf_ids_prod_in_num
                                                .len()
                                                > 0
                                            {
                                                // let mc_factor_num = hc_rule
                                                //     .mc_factor
                                                //     .e_surf_ids_prod_in_num
                                                //     .iter()
                                                //     .map(|e_surf_id| {
                                                //         ct.e_surface_evals[0][*e_surf_id]
                                                //             .cached_eval()
                                                //             * ct.e_surface_evals[0][*e_surf_id]
                                                //                 .cached_eval()
                                                //     })
                                                //     .product::<T>();
                                                let mc_factor_power = Into::<T>::into(
                                                    self.integrand_settings
                                                        .threshold_ct_settings
                                                        .sectoring_settings
                                                        .mc_factor_power,
                                                );
                                                let check_for_absent_e_surfaces_when_building_mc_factor = self
                                                .integrand_settings
                                                .threshold_ct_settings
                                                .sectoring_settings
                                                .check_for_absent_e_surfaces_when_building_mc_factor;

                                                if mc_factor_power > T::zero() {
                                                    let mut mc_factor_num = T::one();
                                                    let mut found_one_factor_in_num = false;

                                                    for (e_surf_id_a, e_surf_id_b) in hc_rule
                                                        .mc_factor
                                                        .e_surf_ids_prod_in_num
                                                        .iter()
                                                    {
                                                        if check_for_absent_e_surfaces_when_building_mc_factor
                                                        && (!self
                                                            .supergraph
                                                            .supergraph_cff
                                                            .cff_expression
                                                            .terms[i_cff]
                                                            .contains_e_surf_id(*e_surf_id_a)
                                                            || !self
                                                                .supergraph
                                                                .supergraph_cff
                                                                .cff_expression
                                                                .terms[i_cff]
                                                                .contains_e_surf_id(*e_surf_id_b))
                                                    {
                                                        continue;
                                                    }
                                                        found_one_factor_in_num = true;
                                                        mc_factor_num *= ((ct.e_surface_evals[0]
                                                            [*e_surf_id_a]
                                                            .cached_eval()
                                                            - ct.e_surface_evals[0][*e_surf_id_b]
                                                                .cached_eval())
                                                            / Into::<T>::into(
                                                                self.settings.kinematics.e_cm
                                                                    * self.settings.kinematics.e_cm
                                                                        as f64,
                                                            ))
                                                        .abs()
                                                        .powf(mc_factor_power);
                                                    }
                                                    let mut mc_factor_den = T::zero();
                                                    let mut n_terms = 0_usize;
                                                    for term in hc_rule
                                                        .mc_factor
                                                        .e_surf_ids_prods_to_sum_in_denom
                                                        .iter()
                                                    {
                                                        let mut mc_factor_den_term = T::one();
                                                        let mut found_one_factor = false;
                                                        for (e_surf_id_a, e_surf_id_b) in
                                                            term.iter()
                                                        {
                                                            if check_for_absent_e_surfaces_when_building_mc_factor
                                                            && (!self
                                                                .supergraph
                                                                .supergraph_cff
                                                                .cff_expression
                                                                .terms[i_cff]
                                                                .contains_e_surf_id(*e_surf_id_a)
                                                                || !self
                                                                    .supergraph
                                                                    .supergraph_cff
                                                                    .cff_expression
                                                                    .terms[i_cff]
                                                                    .contains_e_surf_id(
                                                                        *e_surf_id_b,
                                                                    ))
                                                        {
                                                            continue;
                                                        }
                                                            found_one_factor = true;
                                                            mc_factor_den_term *= ((ct
                                                                .e_surface_evals[0][*e_surf_id_a]
                                                                .cached_eval()
                                                                - ct.e_surface_evals[0]
                                                                    [*e_surf_id_b]
                                                                    .cached_eval())
                                                                / Into::<T>::into(
                                                                    self.settings.kinematics.e_cm
                                                                        * self
                                                                            .settings
                                                                            .kinematics
                                                                            .e_cm
                                                                            as f64,
                                                                ))
                                                            .abs()
                                                            .powf(mc_factor_power);
                                                        }
                                                        if found_one_factor {
                                                            mc_factor_den += mc_factor_den_term;
                                                            n_terms += 1;
                                                        }
                                                    }
                                                    if n_terms > 1 && found_one_factor_in_num {
                                                        if self.settings.general.debug > 5 {
                                                            println!("     | E-surface evaluations for the computation of the MC factor specified as {:?}:\n{}",hc_rule.mc_factor,
                                                            ct.e_surface_evals[0].iter().enumerate().map(|(i,sc)| format!("     |   E-surface #{:-2}: {:+.16e}",i,sc.cached_eval())).collect::<Vec<_>>().join("\n")
                                                        );
                                                        }
                                                        if self.settings.general.debug > 4 {
                                                            println!("     | MC factor specified as {:?} yielded the following mc_factor weight: {:+.16e}",hc_rule.mc_factor, mc_factor_num / mc_factor_den);
                                                        }
                                                    } else {
                                                        if self.settings.general.debug > 4 {
                                                            println!("     | MC factor specified as {:?} yielded no mc_factor (therefore set to one) since not enough e-surfaces of this factor are present in this cFF term.",hc_rule.mc_factor);
                                                        }
                                                    }
                                                    if n_terms > 1 && found_one_factor_in_num {
                                                        mc_factor_num / mc_factor_den
                                                    } else {
                                                        T::one()
                                                    }
                                                } else {
                                                    // Use the min of e_surf_evals, this only make sense if we're comparing only two entities
                                                    assert!(
                                                        hc_rule
                                                            .mc_factor
                                                            .e_surf_ids_prods_to_sum_in_denom
                                                            .len()
                                                            == 2
                                                    );
                                                    let mut min_eval_num = None;
                                                    for (e_surf_id_a, e_surf_id_b) in hc_rule
                                                        .mc_factor
                                                        .e_surf_ids_prod_in_num
                                                        .iter()
                                                    {
                                                        if check_for_absent_e_surfaces_when_building_mc_factor
                                                        && (!self
                                                            .supergraph
                                                            .supergraph_cff
                                                            .cff_expression
                                                            .terms[i_cff]
                                                            .contains_e_surf_id(*e_surf_id_a)
                                                            || !self
                                                                .supergraph
                                                                .supergraph_cff
                                                                .cff_expression
                                                                .terms[i_cff]
                                                                .contains_e_surf_id(*e_surf_id_b))
                                                    {
                                                        continue;
                                                    }
                                                        let t = (ct.e_surface_evals[0]
                                                            [*e_surf_id_a]
                                                            .cached_eval()
                                                            - ct.e_surface_evals[0][*e_surf_id_b]
                                                                .cached_eval())
                                                        .abs();
                                                        if let Some(min_eval) = min_eval_num {
                                                            if t < min_eval {
                                                                min_eval_num = Some(t);
                                                            }
                                                        } else {
                                                            min_eval_num = Some(t);
                                                        }
                                                    }

                                                    if min_eval_num.is_none() {
                                                        if self.settings.general.debug > 4 {
                                                            println!("     | MC factor specified as {:?} yielded no mc_factor (therefore set to one) since not enough e-surfaces of this factor are present in this cFF term.",hc_rule.mc_factor);
                                                        }
                                                        T::one()
                                                    } else {
                                                        let mut min_eval_denom = None;
                                                        for term in hc_rule
                                                            .mc_factor
                                                            .e_surf_ids_prods_to_sum_in_denom
                                                            .iter()
                                                        {
                                                            for (e_surf_id_a, e_surf_id_b) in
                                                                term.iter()
                                                            {
                                                                if check_for_absent_e_surfaces_when_building_mc_factor
                                                                && (!self
                                                                    .supergraph
                                                                    .supergraph_cff
                                                                    .cff_expression
                                                                    .terms[i_cff]
                                                                    .contains_e_surf_id(*e_surf_id_a)
                                                                    || !self
                                                                        .supergraph
                                                                        .supergraph_cff
                                                                        .cff_expression
                                                                        .terms[i_cff]
                                                                        .contains_e_surf_id(
                                                                            *e_surf_id_b,
                                                                        ))
                                                                {
                                                                    continue;
                                                                }
                                                                let t = (ct.e_surface_evals[0]
                                                                    [*e_surf_id_a]
                                                                    .cached_eval()
                                                                    - ct.e_surface_evals[0]
                                                                        [*e_surf_id_b]
                                                                        .cached_eval())
                                                                .abs();
                                                                if let Some(min_eval) =
                                                                    min_eval_denom
                                                                {
                                                                    if t < min_eval {
                                                                        min_eval_denom = Some(t);
                                                                    }
                                                                } else {
                                                                    min_eval_denom = Some(t);
                                                                }
                                                            }
                                                        }

                                                        if min_eval_denom.is_none() {
                                                            if self.settings.general.debug > 4 {
                                                                println!("     | MC factor specified as {:?} yielded no mc_factor (therefore set to one) since not enough e-surfaces of this factor are present in this cFF term.",hc_rule.mc_factor);
                                                            }
                                                            T::one()
                                                        } else {
                                                            if self.settings.general.debug > 5 {
                                                                println!("     | E-surface evaluations for the computation of the MC factor specified as {:?}:\n{}",hc_rule.mc_factor,
                                                                ct.e_surface_evals[0].iter().enumerate().map(|(i,sc)| format!("     |   E-surface #{:-2}: {:+.16e}",i,sc.cached_eval())).collect::<Vec<_>>().join("\n")
                                                            );
                                                            }
                                                            if self.settings.general.debug > 4 {
                                                                println!("     | MC factor specified as {:?} yielded the following mc_factor weight: {:+.16e}",hc_rule.mc_factor, min_eval_num.unwrap() / min_eval_denom.unwrap());
                                                            }
                                                            if min_eval_denom.unwrap()
                                                                < min_eval_num.unwrap()
                                                            {
                                                                T::one()
                                                            } else {
                                                                T::zero()
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                // MC factor is absent then here
                                                T::one()
                                            };
                                            if mc_factor == T::zero() {
                                                keep_this_one_ct = false;
                                                if self.settings.general.debug > 3 {
                                                    reason_for_this_ct = format!("it was found in a hard-coded rule with mc_factor {:?} evaluating to zero.",hc_rule.mc_factor);
                                                }
                                            }
                                        } else {
                                            keep_this_one_ct = false;
                                            if self.settings.general.debug > 3 {
                                                reason_for_this_ct = format!("{}", "it was found to be explicitly disabled in a hard-coded rule.");
                                            }
                                        }
                                    } else {
                                        if self.settings.general.debug > 3 {
                                            println!(
                                            "     | Did not find any hard-coded rule applying to this counterterm with sector signature {:?} within cut #{}",
                                            this_ct_sector_signature,i_cut
                                        )
                                        }
                                    }
                                }
                                if !keep_this_one_ct {
                                    if self.settings.general.debug > 3 {
                                        println!(
                                            "     | cFF Evaluation #{} : Ignoring CT for {} E-surface {} solved in {} with sector signature {} because {}",
                                            format!("{}", i_cff).green(),
                                            format!(
                                                "{}|{}|{}",
                                                if side == LEFT { "L" } else { "R" },
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
                                            format!("{:?}", this_ct_sector_signature).blue(),
                                            reason_for_this_ct
                                        );
                                    }
                                    continue;
                                } else {
                                    if rule_found.is_some() {
                                        cts[side].push((ct, mc_factor));
                                        continue;
                                    }
                                }
                            }

                            if self
                                .integrand_settings
                                .threshold_ct_settings
                                .sectoring_settings
                                .accept_all
                            {
                                if self.settings.general.debug > 0 {
                                    println!(
                                        "{}",
                                        format!("{}", "   > User forced accepting all CTs")
                                            .bold()
                                            .red()
                                    );
                                }
                                let mut keep_this_one_ct = true;
                                let mut reason_for_this_ct = format!("{}", "");
                                for (sig_i_cut, cut_cache) in cache.cut_caches.iter().enumerate() {
                                    let cut_str = if self.settings.general.debug > 3 {
                                        format!(
                                            "{}({})",
                                            format!("#{}", sig_i_cut).blue(),
                                            format!(
                                                "{}",
                                                self.supergraph.cuts[sig_i_cut]
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
                                            .blue()
                                        )
                                    } else {
                                        format!("{}", "")
                                    };
                                    if cut_cache.cut_sg_e_surf_id == ct.e_surf_id
                                        && self
                                            .integrand_settings
                                            .threshold_ct_settings
                                            .sectoring_settings
                                            .anti_select_threshold_against_observable
                                    {
                                        if this_ct_sector_signature[sig_i_cut] == CUT_ACTIVE {
                                            keep_this_one_ct = false;
                                            if self.settings.general.debug > 3 {
                                                reason_for_this_ct = format!("it contains the following cut {} which is the threshold itself and selected by the observable, so anti-selection of threshold kicks in.",
                                                    cut_str
                                                );
                                            }
                                            break;
                                        }
                                    }
                                }

                                if keep_this_one_ct {
                                    cts[side].push((ct, T::one()));
                                } else {
                                    if self.settings.general.debug > 3 {
                                        println!(
                                            "     | cFF Evaluation #{} : Ignoring CT for {} E-surface {} solved in {} with sector signature {} because {}",
                                            format!("{}", i_cff).green(),
                                            format!(
                                                "{}|{}|{}",
                                                if side == LEFT { "L" } else { "R" },
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
                                            format!("{:?}", this_ct_sector_signature).blue(),
                                            reason_for_this_ct
                                        );
                                    }
                                }

                                continue;
                            } else if self
                                .integrand_settings
                                .threshold_ct_settings
                                .sectoring_settings
                                .sector_based_analysis
                                || self
                                    .integrand_settings
                                    .threshold_ct_settings
                                    .sectoring_settings
                                    .force_one_loop_ct_in_soft_sector
                                || self
                                    .integrand_settings
                                    .threshold_ct_settings
                                    .sectoring_settings
                                    .always_solve_cts_in_all_amplitude_loop_indices
                            {
                                let amplitude_for_sides = [
                                    &self.supergraph.cuts[i_cut].left_amplitude,
                                    &self.supergraph.cuts[i_cut].right_amplitude,
                                ];

                                let mut keep_this_one_ct = true;
                                let mut reason_for_this_ct = format!("{}", "");

                                let n_loop_ct =
                                    ct.loop_indices_solved.0.len() + ct.loop_indices_solved.1.len();
                                let was_projected = if ct.ct_level == SUPERGRAPH_LEVEL_CT {
                                    ct.loop_indices_solved.0.len()
                                        < self.supergraph.cuts[i_cut].left_amplitude.n_loop
                                        || ct.loop_indices_solved.1.len()
                                            < self.supergraph.cuts[i_cut].right_amplitude.n_loop
                                } else {
                                    if side == LEFT {
                                        ct.loop_indices_solved.0.len()
                                            < self.supergraph.cuts[i_cut].left_amplitude.n_loop
                                    } else {
                                        ct.loop_indices_solved.1.len()
                                            < self.supergraph.cuts[i_cut].right_amplitude.n_loop
                                    }
                                };

                                let mut loop_indices_in_this_amplitude = vec![];
                                for lmb_e in amplitude_for_sides[side].lmb_edges.iter() {
                                    for (i_s, s) in self.supergraph.supergraph_cff.edges[lmb_e.id]
                                        .signature
                                        .0
                                        .iter()
                                        .enumerate()
                                    {
                                        if *s != 0 && !loop_indices_in_this_amplitude.contains(&i_s)
                                        {
                                            loop_indices_in_this_amplitude.push(i_s);
                                        }
                                    }
                                }

                                let mut loop_indices_available_for_subtraction_in_this_sector =
                                    vec![];

                                for (sig_i_cut, cut_cache) in cache.cut_caches.iter().enumerate() {
                                    let cut_str = if self.settings.general.debug > 3 {
                                        format!(
                                            "{}({})",
                                            format!("#{}", sig_i_cut).blue(),
                                            format!(
                                                "{}",
                                                self.supergraph.cuts[sig_i_cut]
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
                                            .blue()
                                        )
                                    } else {
                                        format!("{}", "")
                                    };

                                    if this_ct_sector_signature[sig_i_cut] == CUT_ABSENT {
                                        if cut_cache.cut_sg_e_surf_id == ct.e_surf_id {
                                            keep_this_one_ct = false;
                                            if self.settings.general.debug > 3 {
                                                reason_for_this_ct = format!("it contains the following cut {} which is the threshold itself and absent for this cFF term so it is removed",
                                                    cut_str
                                                );
                                            }
                                            break;
                                        }
                                    } else if this_ct_sector_signature[sig_i_cut] == CUT_ACTIVE {
                                        if cut_cache.cut_sg_e_surf_id == ct.e_surf_id
                                            && self
                                                .integrand_settings
                                                .threshold_ct_settings
                                                .sectoring_settings
                                                .anti_select_threshold_against_observable
                                        {
                                            keep_this_one_ct = false;
                                            if self.settings.general.debug > 3 {
                                                reason_for_this_ct = format!("it contains the following cut {} which is the threshold itself and selected by the observable, so anti-selection of threshold kicks in.",
                                                    cut_str
                                                );
                                            }
                                            break;
                                        }
                                    } else {
                                        if this_ct_sector_signature[sig_i_cut] == CUT_INACTIVE {
                                            for li in &[0, 1, 2] {
                                                if loop_indices_available_for_subtraction_in_this_sector.contains(li) || !loop_indices_in_this_amplitude.contains(li) {
                                                    continue;
                                                }
                                                for (i_edge, _flip) in self.supergraph.cuts
                                                    [sig_i_cut]
                                                    .cut_edge_ids_and_flip
                                                    .iter()
                                                {
                                                    if self.supergraph.supergraph_cff.edges[*i_edge]
                                                        .signature
                                                        .0[*li]
                                                        != 0
                                                    {
                                                        loop_indices_available_for_subtraction_in_this_sector.push(*li);
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if !keep_this_one_ct {
                                        break;
                                    }
                                }

                                // Custom rules removing available indices for the vertical cuts
                                // left vertical cut
                                if this_ct_sector_signature[0] == CUT_ACTIVE
                                    && loop_indices_available_for_subtraction_in_this_sector
                                        .contains(&0)
                                {
                                    loop_indices_available_for_subtraction_in_this_sector.remove(
                                        loop_indices_available_for_subtraction_in_this_sector
                                            .iter()
                                            .position(|&li| li == 0)
                                            .unwrap(),
                                    );
                                }
                                // right vertical cut
                                if this_ct_sector_signature[1] == CUT_ACTIVE
                                    && loop_indices_available_for_subtraction_in_this_sector
                                        .contains(&1)
                                {
                                    loop_indices_available_for_subtraction_in_this_sector.remove(
                                        loop_indices_available_for_subtraction_in_this_sector
                                            .iter()
                                            .position(|&li| li == 1)
                                            .unwrap(),
                                    );
                                }
                                // middle vertical cut
                                if this_ct_sector_signature[8] == CUT_ACTIVE
                                    && loop_indices_available_for_subtraction_in_this_sector
                                        .contains(&2)
                                {
                                    loop_indices_available_for_subtraction_in_this_sector.remove(
                                        loop_indices_available_for_subtraction_in_this_sector
                                            .iter()
                                            .position(|&li| li == 2)
                                            .unwrap(),
                                    );
                                }

                                if keep_this_one_ct {
                                    if self
                                        .integrand_settings
                                        .threshold_ct_settings
                                        .sectoring_settings
                                        .always_solve_cts_in_all_amplitude_loop_indices
                                    {
                                        if !loop_indices_in_this_amplitude
                                            .iter()
                                            .all(|lia| loop_indices_solved.contains(lia))
                                        {
                                            keep_this_one_ct = false;
                                            if self.settings.general.debug > 3 {
                                                reason_for_this_ct = format!("it is a counterterm solved only in loop_indices {:?}, but the soft sector analysis requires to keep only CTs solved in ALL loop momenta ({:?}).",
                                                        loop_indices_solved, loop_indices_in_this_amplitude
                                                    );
                                            }
                                        }
                                    }

                                    let mut force_projected_one_loop_cts = false;
                                    if self
                                        .integrand_settings
                                        .threshold_ct_settings
                                        .sectoring_settings
                                        .force_one_loop_ct_in_soft_sector
                                    {
                                        if self
                                            .integrand_settings
                                            .threshold_ct_settings
                                            .sectoring_settings
                                            .always_solve_cts_in_all_amplitude_loop_indices
                                        {
                                            if self.settings.general.debug > 4 {
                                                println!("     | User forced CTs to always be solved in the complete set of amplitude loop indices.");
                                            }
                                        } else {
                                            for combinations_identifying_soft_sectors in [
                                                [
                                                    (0, [CUT_ACTIVE, CUT_ABSENT]),
                                                    (3, [CUT_ACTIVE, CUT_ABSENT]),
                                                    (2, [CUT_ACTIVE, CUT_ABSENT]),
                                                    (8, [CUT_ACTIVE, CUT_ABSENT]),
                                                ],
                                                [
                                                    (1, [CUT_ACTIVE, CUT_ABSENT]),
                                                    (4, [CUT_ACTIVE, CUT_ABSENT]),
                                                    (5, [CUT_ACTIVE, CUT_ABSENT]),
                                                    (8, [CUT_ACTIVE, CUT_ABSENT]),
                                                ],
                                            ] {
                                                if combinations_identifying_soft_sectors.iter().all(
                                                    |(soft_i_cut, selected_status)| {
                                                        selected_status.contains(
                                                            &this_ct_sector_signature[*soft_i_cut],
                                                        )
                                                    },
                                                ) {
                                                    force_projected_one_loop_cts = true;
                                                }
                                            }
                                            if self.settings.general.debug > 4 {
                                                if force_projected_one_loop_cts {
                                                    println!("     | Soft analysis did find a soft sector and all two-loop CTs will be required to be solved in projected spaces.");
                                                } else {
                                                    println!("     | Soft analysis found no soft sector and CTs will be required to be solved in the full set of amplitude loop indices.");
                                                }
                                            }
                                        }

                                        if force_projected_one_loop_cts {
                                            if n_loop_ct > 1
                                                && loop_indices_in_this_amplitude
                                                    .iter()
                                                    .all(|lia| loop_indices_solved.contains(lia))
                                            {
                                                keep_this_one_ct = false;
                                                if self.settings.general.debug > 3 {
                                                    reason_for_this_ct = format!("it is a counterterm solved in the loop_indices {:?} which are ALL loop momenta ({:?}), but the soft sector analysis requires to keep only CTs solved the projected space.",
                                                        loop_indices_solved, loop_indices_in_this_amplitude
                                                    );
                                                }
                                            }
                                        }
                                    }

                                    if keep_this_one_ct && !force_projected_one_loop_cts {
                                        if self.settings.general.debug > 4 {
                                            println!("     | Active loop indices for this CT: {:?}, and present in the amplitude being subtracted: {:?}",loop_indices_available_for_subtraction_in_this_sector,loop_indices_in_this_amplitude);
                                        }

                                        if n_loop_ct > 1 {
                                            if !loop_indices_solved.iter().all(|lis| loop_indices_available_for_subtraction_in_this_sector.contains(lis)) {
                                                keep_this_one_ct = false;
                                                if self.settings.general.debug > 3 {
                                                    reason_for_this_ct = format!("it is a two-loop e-surface in the loop_indices {:?} that are not all available (availabled ones: {:?}).",
                                                        loop_indices_solved, loop_indices_available_for_subtraction_in_this_sector
                                                    );
                                                }
                                            }
                                        }

                                        if was_projected {
                                            if loop_indices_in_this_amplitude.iter().all(|lis| loop_indices_available_for_subtraction_in_this_sector.contains(lis)) {
                                                keep_this_one_ct = false;
                                                if self.settings.general.debug > 3 {
                                                    reason_for_this_ct = format!("it is a one-loop projected e-surface of an amplitude with loop indices {:?} that are all available (availabled ones: {:?}), so the two-loop CT should be used instead",
                                                        loop_indices_in_this_amplitude, loop_indices_available_for_subtraction_in_this_sector
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }

                                if keep_this_one_ct {
                                    cts[side].push((ct, T::one()));
                                } else {
                                    if self.settings.general.debug > 3 {
                                        println!(
                                            "     | cFF Evaluation #{} : Ignoring CT for {} E-surface {} solved in {} with sector signature {} because {}",
                                            format!("{}", i_cff).green(),
                                            format!(
                                                "{}|{}|{}",
                                                if side == LEFT { "L" } else { "R" },
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
                                            format!("{:?}", this_ct_sector_signature).blue(),
                                            reason_for_this_ct
                                        );
                                    }
                                }

                                continue;
                            } else {
                                // WARNING: Do not correlate signature of CTs with event as it breaks the PV being zero
                                let (keep_this_ct, reason) = if !self
                                    .integrand_settings
                                    .threshold_ct_settings
                                    .sectoring_settings
                                    .correlate_event_sector_with_ct_sector
                                    || this_ct_sector_signature == cut_sector_signature
                                {
                                    let mut keep_this_one_ct = true;
                                    let mut reason_for_this_ct = format!("{}", "");

                                    let mut min_active_pinched_e_surf_sq_eval_in_solved_subspace =
                                        Option::<(usize, T)>::None;
                                    let mut
                                    min_active_pinched_e_surf_sq_eval_in_projected_out_subspace =
                                        Option::<(usize, T)>::None;
                                    let mut
                                    min_inactive_non_pinched_e_surf_sq_eval_in_solved_subspace =
                                        Option::<(usize, T)>::None;
                                    let mut
                                    min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace =
                                        Option::<(usize, T)>::None;

                                    for (sig_i_cut, cut_cache) in
                                        cache.cut_caches.iter().enumerate()
                                    {
                                        let cut_str = if self.settings.general.debug > 3 {
                                            format!(
                                                "{}({})",
                                                format!("#{}", sig_i_cut).blue(),
                                                format!(
                                                    "{}",
                                                    self.supergraph.cuts[sig_i_cut]
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
                                                .blue()
                                            )
                                        } else {
                                            format!("{}", "")
                                        };

                                        if this_ct_sector_signature[sig_i_cut] == CUT_ABSENT {
                                            if cut_cache.cut_sg_e_surf_id == ct.e_surf_id {
                                                keep_this_one_ct = false;
                                                if self.settings.general.debug > 3 {
                                                    reason_for_this_ct = format!("it contains the following cut {} which is the threshold itself and absent for this cFF term so it is removed",
                                                        cut_str
                                                    );
                                                }
                                                break;
                                            }
                                        } else if this_ct_sector_signature[sig_i_cut] == CUT_ACTIVE
                                        {
                                            if cut_cache.cut_sg_e_surf_id == ct.e_surf_id
                                                && self
                                                    .integrand_settings
                                                    .threshold_ct_settings
                                                    .sectoring_settings
                                                    .anti_select_threshold_against_observable
                                            {
                                                keep_this_one_ct = false;
                                                if self.settings.general.debug > 3 {
                                                    reason_for_this_ct = format!("it contains the following cut {} which is the threshold itself and selected by the observable, so anti-selection of threshold kicks in.",
                                                        cut_str
                                                    );
                                                }
                                                break;
                                            }

                                            if ct
                                                .e_surface_analysis
                                                .pinched_e_surf_ids_active_solved_subspace
                                                .contains(&cut_cache.cut_sg_e_surf_id)
                                            {
                                                if let Some((curr_min_i_cut, curr_min_sq_eval)) =
                                                    min_active_pinched_e_surf_sq_eval_in_solved_subspace
                                                {
                                                    if ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id]
                                                        .cached_eval()
                                                        * ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id]
                                                        .cached_eval()
                                                        < curr_min_sq_eval
                                                    {
                                                        min_active_pinched_e_surf_sq_eval_in_solved_subspace = Some( (sig_i_cut, ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()) );
                                                    }
                                                } else {
                                                    min_active_pinched_e_surf_sq_eval_in_solved_subspace = Some( (sig_i_cut, ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()) );
                                                }
                                            }
                                            if ct
                                                .e_surface_analysis
                                                .pinched_e_surf_ids_active_in_projected_out_subspace
                                                .contains(&cut_cache.cut_sg_e_surf_id)
                                            {
                                                if let Some( (curr_min_i_cut, curr_min_sq_eval) ) = min_active_pinched_e_surf_sq_eval_in_projected_out_subspace {
                                                    if ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval() < curr_min_sq_eval {
                                                        min_active_pinched_e_surf_sq_eval_in_projected_out_subspace = Some( (sig_i_cut, ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()) );
                                                    }
                                                } else {
                                                    min_active_pinched_e_surf_sq_eval_in_projected_out_subspace = Some( (sig_i_cut, ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()) );
                                                }
                                            }
                                        } else if this_ct_sector_signature[sig_i_cut]
                                            == CUT_INACTIVE
                                        {
                                            if ct
                                                .e_surface_analysis
                                                .non_pinched_e_surf_ids_active_solved_subspace
                                                .contains(&cut_cache.cut_sg_e_surf_id)
                                            {
                                                if let Some((curr_min_i_cut, curr_min_sq_eval)) =
                                                    min_inactive_non_pinched_e_surf_sq_eval_in_solved_subspace
                                                {
                                                    if ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id]
                                                        .cached_eval()
                                                        * ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id]
                                                        .cached_eval()
                                                        < curr_min_sq_eval
                                                    {
                                                        min_inactive_non_pinched_e_surf_sq_eval_in_solved_subspace = Some( (sig_i_cut, ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()) );
                                                    }
                                                } else {
                                                    min_inactive_non_pinched_e_surf_sq_eval_in_solved_subspace = Some( (sig_i_cut, ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()) );
                                                }
                                            }
                                            if ct
                                                .e_surface_analysis
                                                .non_pinched_e_surf_ids_active_in_projected_out_subspace
                                                .contains(&cut_cache.cut_sg_e_surf_id)
                                            {
                                                if let Some( (curr_min_i_cut, curr_min_sq_eval) ) = min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace {
                                                    if ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval() < curr_min_sq_eval {
                                                        min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace = Some( (sig_i_cut, ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()) );
                                                    }
                                                } else {
                                                    min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace = Some( (sig_i_cut, ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()*ct.e_surface_evals[0][cut_cache.cut_sg_e_surf_id].cached_eval()) );
                                                }
                                            }
                                        }
                                    }

                                    if keep_this_one_ct {
                                        let n_loop_ct = ct.loop_indices_solved.0.len()
                                            + ct.loop_indices_solved.1.len();
                                        let was_projected = if ct.ct_level == SUPERGRAPH_LEVEL_CT {
                                            ct.loop_indices_solved.0.len()
                                                < self.supergraph.cuts[i_cut].left_amplitude.n_loop
                                                || ct.loop_indices_solved.1.len()
                                                    < self.supergraph.cuts[i_cut]
                                                        .right_amplitude
                                                        .n_loop
                                        } else {
                                            if side == LEFT {
                                                ct.loop_indices_solved.0.len()
                                                    < self.supergraph.cuts[i_cut]
                                                        .left_amplitude
                                                        .n_loop
                                            } else {
                                                ct.loop_indices_solved.1.len()
                                                    < self.supergraph.cuts[i_cut]
                                                        .right_amplitude
                                                        .n_loop
                                            }
                                        };

                                        if self.settings.general.debug > 4 {
                                            println!("Min active pinched e_surf in solved subspace: {}",
                                                if let Some((
                                                    min_i_cut,
                                                    _eval,
                                                )) = min_active_pinched_e_surf_sq_eval_in_solved_subspace {
                                                    format!("from cut {}({}), eval = {:+16.e}",
                                                        format!("#{}",min_i_cut).blue(),
                                                        format!("{}",self.supergraph.cuts[min_i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                        ct.e_surface_evals[0][cache.cut_caches[min_i_cut].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64)
                                                    ).normal()
                                                } else {
                                                    format!("{}","None").blue()
                                                }
                                            );
                                            println!("Min active pinched e_surf in projected out subspace: {}",
                                                if let Some((
                                                    min_i_cut,
                                                    _eval,
                                                )) = min_active_pinched_e_surf_sq_eval_in_projected_out_subspace {
                                                    format!("from cut {}({}), eval = {:+16.e}",
                                                        format!("#{}",min_i_cut).blue(),
                                                        format!("{}",self.supergraph.cuts[min_i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                        ct.e_surface_evals[0][cache.cut_caches[min_i_cut].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64)
                                                    ).normal()
                                                } else {
                                                    format!("{}","None").blue()
                                                }
                                            );
                                            println!("Min inactive non-pinched e_surf in solved subspace: {}",
                                                if let Some((
                                                    min_i_cut,
                                                    _eval,
                                                )) = min_inactive_non_pinched_e_surf_sq_eval_in_solved_subspace {
                                                    format!("from cut {}({}), eval = {:+16.e}",
                                                        format!("#{}",min_i_cut).blue(),
                                                        format!("{}",self.supergraph.cuts[min_i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                        ct.e_surface_evals[0][cache.cut_caches[min_i_cut].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64)
                                                    ).normal()
                                                } else {
                                                    format!("{}","None").blue()
                                                }
                                            );
                                            println!("Min inactive non-pinched e_surf in projected out subspace: {}",
                                                if let Some((
                                                    min_i_cut,
                                                    _eval,
                                                )) = min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace {
                                                    format!("from cut {}({}), eval = {:+16.e}",
                                                        format!("#{}",min_i_cut).blue(),
                                                        format!("{}",self.supergraph.cuts[min_i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                        ct.e_surface_evals[0][cache.cut_caches[min_i_cut].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64)
                                                    ).normal()
                                                } else {
                                                    format!("{}","None").blue()
                                                }
                                            );
                                        }
                                        // First perform general checks that always apply
                                        if let Some((
                                            min_i_cut,
                                            actual_min_active_pinched_e_surf_sq_eval_in_solved_subspace,
                                        )) = min_active_pinched_e_surf_sq_eval_in_solved_subspace
                                        {
                                            if actual_min_active_pinched_e_surf_sq_eval_in_solved_subspace < ct.e_surface_evals[0][ct.e_surf_id].cached_eval()*ct.e_surface_evals[0][ct.e_surf_id].cached_eval() {
                                                keep_this_one_ct = false;
                                                reason_for_this_ct = if self.settings.general.debug > 3 {
                                                    format!("it is a one-loop CT which contains an active pinched surface corresponding to cut {}({}) within the solved space with an eval of {:+16.e} wich is smaller in abs. value than self CT esurf eval of {:+16e}.",
                                                        format!("#{}",min_i_cut).blue(),
                                                        format!("{}",self.supergraph.cuts[min_i_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                        ct.e_surface_evals[0][cache.cut_caches[min_i_cut].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64),
                                                        ct.e_surface_evals[0][ct.e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64)
                                                    )
                                                } else {
                                                    format!("{}","")
                                                };
                                            }
                                        }
                                        // Perform further checks if this a projected out one-loop from two-loop CTs or a two-loop CT itself and not yet dismissed
                                        if keep_this_one_ct {
                                            if was_projected {
                                                let mut min_pinched = min_active_pinched_e_surf_sq_eval_in_solved_subspace;
                                                if let Some((
                                                    min_i_pinched_cut,
                                                    actual_min_active_pinched_e_surf_sq_eval_in_projected_out_subspace,
                                                )) = min_active_pinched_e_surf_sq_eval_in_projected_out_subspace {
                                                    if let Some((curr_i, curr_min)) = min_pinched {
                                                        if actual_min_active_pinched_e_surf_sq_eval_in_projected_out_subspace < curr_min {
                                                            min_pinched = Some((min_i_pinched_cut, actual_min_active_pinched_e_surf_sq_eval_in_projected_out_subspace))
                                                        }
                                                    } else {
                                                        min_pinched = Some((min_i_pinched_cut, actual_min_active_pinched_e_surf_sq_eval_in_projected_out_subspace))
                                                    }
                                                }
                                                let mut min_non_pinched = min_inactive_non_pinched_e_surf_sq_eval_in_solved_subspace;
                                                if let Some((
                                                    min_i_non_pinched_cut,
                                                    actual_min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace,
                                                )) = min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace {
                                                    if let Some((curr_i, curr_min)) = min_non_pinched {
                                                        if actual_min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace < curr_min {
                                                            min_non_pinched = Some((min_i_non_pinched_cut, actual_min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace))
                                                        }
                                                    } else {
                                                        min_non_pinched = Some((min_i_non_pinched_cut, actual_min_inactive_non_pinched_e_surf_sq_eval_in_projected_out_subspace))
                                                    }
                                                }
                                                // We want to keep the projected one-loop CT only when the two-loop CT would be inactivated, which is only when there is a pinched surface anywhere that is
                                                // stronger than the strongest between a non-pinched E-surface anywhere and itself
                                                if let Some((i_pinched, min_pinched_eval)) =
                                                    min_pinched
                                                {
                                                    if let Some((
                                                        i_non_pinched,
                                                        min_non_pinched_eval,
                                                    )) = min_non_pinched
                                                    {
                                                        if min_pinched_eval > min_non_pinched_eval {
                                                            keep_this_one_ct = false;
                                                            reason_for_this_ct = if self
                                                                .settings
                                                                .general
                                                                .debug
                                                                > 3
                                                            {
                                                                format!("it is a one-loop projected CT which contains, within both solved and projected space, a weaker active pinched surface there {}({}) with eval {:+16e} than the inactive non-pinched surface {}({}) with eval {:+16e}.",
                                                                    format!("#{}",i_pinched).blue(),
                                                                    format!("{}",self.supergraph.cuts[i_pinched].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                                    ct.e_surface_evals[0][cache.cut_caches[i_pinched].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64),
                                                                    format!("#{}",i_non_pinched).blue(),
                                                                    format!("{}",self.supergraph.cuts[i_non_pinched].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                                    ct.e_surface_evals[0][cache.cut_caches[i_non_pinched].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64)
                                                                )
                                                            } else {
                                                                format!("{}", "")
                                                            };
                                                        }
                                                    } else {
                                                        keep_this_one_ct = false;
                                                        reason_for_this_ct = if self
                                                            .settings
                                                            .general
                                                            .debug
                                                            > 3
                                                        {
                                                            format!("it is a one-loop projected CT which contains, within both solved and projected space, an active pinched surface there {}({}) with eval {:+16e} and no non-pinched one, so we will use the two-loop CT.",
                                                                format!("#{}",i_pinched).blue(),
                                                                format!("{}",self.supergraph.cuts[i_pinched].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                                ct.e_surface_evals[0][cache.cut_caches[i_pinched].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64),
                                                            )
                                                        } else {
                                                            format!("{}", "")
                                                        };
                                                    }
                                                } else {
                                                    keep_this_one_ct = false;
                                                    reason_for_this_ct = if self
                                                        .settings
                                                        .general
                                                        .debug
                                                        > 3
                                                    {
                                                        format!("{}","it is a projected one-loop CT which does not contain any active pinched surface anywhere, so the two-loop CT will be used instead.")
                                                    } else {
                                                        format!("{}", "")
                                                    };
                                                }
                                            } else if n_loop_ct > 1 {
                                                // The idea is to always keep the two-loop CT except when the strongest non-pinched E-surface and itself within the solved space
                                                // is weaker than the strongest pinched E-surface within the solved space.
                                                if let Some((
                                                    min_i_pinched_cut,
                                                    actual_min_active_pinched_e_surf_sq_eval_in_solved_subspace,
                                                )) = min_active_pinched_e_surf_sq_eval_in_solved_subspace {
                                                    if let Some((
                                                        min_i_non_pinched_cut,
                                                        actual_min_inactive_non_pinched_e_surf_sq_eval_in_solved_subspace,
                                                    )) = min_inactive_non_pinched_e_surf_sq_eval_in_solved_subspace {
                                                        if actual_min_inactive_non_pinched_e_surf_sq_eval_in_solved_subspace > actual_min_active_pinched_e_surf_sq_eval_in_solved_subspace {
                                                            keep_this_one_ct = false;
                                                            reason_for_this_ct = if self.settings.general.debug > 3 {format!("it is a two-loop CT which contains, within the solved space, a stronger active pinched surface {}({}) with eval {:+16e} than min(inactive non-pinched surface {}({}) with eval {:+16e}, self eval {:+16e}).",
                                                                format!("#{}",min_i_pinched_cut).blue(),
                                                                format!("{}",self.supergraph.cuts[min_i_pinched_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                                ct.e_surface_evals[0][cache.cut_caches[min_i_pinched_cut].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64),
                                                                format!("#{}",min_i_non_pinched_cut).blue(),
                                                                format!("{}",self.supergraph.cuts[min_i_non_pinched_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                                ct.e_surface_evals[0][cache.cut_caches[min_i_non_pinched_cut].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64),
                                                                ct.e_surface_evals[0][ct.e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64)
                                                            )} else {
                                                                format!("{}","")
                                                            };
                                                        }
                                                    } else {
                                                        keep_this_one_ct = true;
                                                        // keep_this_one_ct = false;
                                                        // reason_for_this_ct = if self.settings.general.debug > 3 {format!("it is a two-loop CT which contains, within the solved space, a stronger active pinched surface there {}({}) with eval {:+16e} than self eval {:+16e}).",
                                                        //     format!("#{}",min_i_pinched_cut).blue(),
                                                        //     format!("{}",self.supergraph.cuts[min_i_pinched_cut].cut_edge_ids_and_flip.iter().map(|(id, flip)| if *flip > 0 {format!("+{}", id)} else {format!("-{}", id)}).collect::<Vec<_>>().join("|")).blue(),
                                                        //     ct.e_surface_evals[0][cache.cut_caches[min_i_pinched_cut].cut_sg_e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64),
                                                        //     ct.e_surface_evals[0][ct.e_surf_id].cached_eval()/Into::<T>::into(self.settings.kinematics.e_cm as f64)
                                                        // )} else {
                                                        //     format!("{}","")
                                                        // };
                                                    }

                                                }
                                            }
                                        }
                                    }

                                    (keep_this_one_ct, reason_for_this_ct)
                                } else {
                                    (
                                        false,
                                        if self.settings.general.debug > 3 {
                                            format!(
                                                "it has an incompatible sector signatue: {}",
                                                format!("{:?}", this_ct_sector_signature).red()
                                            )
                                        } else {
                                            format!("{}", "")
                                        },
                                    )
                                };
                                if keep_this_ct {
                                    cts[side].push((ct, T::one()));
                                } else {
                                    if self.settings.general.debug > 3 {
                                        println!(
                                            "     | cFF Evaluation #{} : Ignoring CT for {} E-surface {} solved in {} with sector signature {} because {}",
                                            format!("{}", i_cff).green(),
                                            format!(
                                                "{}|{}|{}",
                                                if side == LEFT { "L" } else { "R" },
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
                                            format!("{:?}", this_ct_sector_signature).blue(),
                                            reason
                                        );
                                    }
                                }
                            }
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
            if !original_event_did_pass_selection {
                this_cff_term_contribution = T::zero();
            }
            cff_res[i_cff] += this_cff_term_contribution;

            // Now include counterterms
            let mut cts_sum_for_this_term = Complex::new(T::zero(), T::zero());
            for ct_side in [LEFT, RIGHT] {
                let other_side = if ct_side == LEFT { RIGHT } else { LEFT };
                for (ct, ct_mc_factor) in cts[ct_side].iter() {
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
                        * ct.h_function_wgt
                        * ct_mc_factor;

                    let mut im_ct_weight = if let Some(i_ct) = &ct.integrated_ct {
                        ct_numerator_wgt
                            * ct_e_product.inv()
                            * ct_cff_eval
                            * i_ct.e_surf_residue
                            * i_ct.adjusted_sampling_jac
                            * i_ct.h_function_wgt
                            * ct_mc_factor
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
                        let mut this_ct_sector_signature = ct.ct_sector_signature.clone();
                        for (i_ss, ss) in this_ct_sector_signature.iter_mut().enumerate() {
                            if cut_sector_signature[i_ss] == CUT_ABSENT {
                                *ss = CUT_ABSENT;
                            }
                        }
                        println!(
                            "   > cFF Evaluation #{} : CT for {} E-surface {} solved in {} with sector signature {} {} = {} + i {}",
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
                            format!("{:?}", this_ct_sector_signature).blue(),
                            format!("and MC factor weight: {:+.16e} x ({:+.16e} + i {:+.16e})",ct_mc_factor, re_ct_weight/ct_mc_factor, im_ct_weight/ct_mc_factor ),
                            format!("{:+.e}",re_ct_weight).green(),
                            format!("{:+.e}",im_ct_weight).green(),
                        );
                    }

                    cts_sum_for_this_term += Complex::new(re_ct_weight, im_ct_weight);
                }
            }

            // Now implement the cross terms
            let mut ct_im_squared_weight_for_this_term = T::zero();
            for (left_ct, left_mc_factor) in cts[LEFT].iter() {
                if left_ct.ct_level == SUPERGRAPH_LEVEL_CT {
                    continue;
                }
                for (right_ct, right_mc_factor) in cts[RIGHT].iter() {
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
                                * left_mc_factor
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
                                * right_mc_factor
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
                                "   > cFF Evaluation #{} : CT for {} E-surfaces ({}) x ({}) {}: {} + i {}",
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
                                format!("and MC factor weights = ({:+.16e}) x ({:+.16e})",left_mc_factor,right_mc_factor),
                                format!("{:+.e}",ct_weight.re).green(),
                                format!("{:+.e}",ct_weight.im).green()
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
        if original_event_did_pass_selection {
            self.event_manager
                .event_buffer
                .last_mut()
                .unwrap()
                .integrand =
                Complex::new(cut_res.re.to_f64().unwrap(), cut_res.im.to_f64().unwrap());
        }
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

        let mut computational_cache = TriBoxTriCFFSectoredComputationCache::default();

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
        computational_cache.sampling_xs =
            xs.iter().map(|x| T::to_f64(x).unwrap()).collect::<Vec<_>>();

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

        self.fill_cache(&loop_momenta, &mut computational_cache);

        for i_cut in 0..self.supergraph.cuts.len() {
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

impl HasIntegrand for TriBoxTriCFFSectoredIntegrand {
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

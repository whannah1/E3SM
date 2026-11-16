/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Config.hpp"
#ifdef HOMME_ENABLE_COMPOSE

#include "ComposeTransportImpl.hpp"

namespace Homme {

// For hv_q_correction=1: compute the weak Laplacian of the layer-midpoint
// pressure of the levels Q currently lives on. These are the same layers CEDR
// uses: divdp (dp_star) if tracer and remap steps are independent, otherwise
// dp3d(np1). The pressure doesn't change during the HV subcycles, so this is
// done once per call to advance_hypervis_scalar.
static void
calc_hv_q_lap_p (const ComposeTransportImpl& cti, const int np1) {
  using CTI = ComposeTransportImpl;
  const auto lap_p = cti.m_data.hv_q_lap_p;
  const auto pbuf = cti.m_data.buf1o[0];
  const auto dp3d = cti.m_state.m_dp3d;
  const auto divdp = cti.m_derived.m_divdp;
  const bool use_divdp = cti.m_data.independent_time_steps;
  const Real ptop = cti.m_hvcoord.hybrid_ai0*cti.m_hvcoord.ps0;
  const auto sphere_ops = cti.m_sphere_ops;
  const auto tu_ne = cti.m_tu_ne;
  const auto f = KOKKOS_LAMBDA (const CTI::MT& team) {
    KernelVariables kv(team, tu_ne);
    const auto ie = kv.ie;
    const CTI::SNlev p(Homme::subview(pbuf, kv.team_idx).data());
    const auto scan = [&] (const int i, const int j) {
      Real p_int = ptop;
      Real p_mid = ptop;
      for (int k = 0; k < CTI::num_phys_lev; ++k) {
        const int kp = k / CTI::packn, ks = k % CTI::packn;
        const Real dp = use_divdp ? divdp(ie,i,j,kp)[ks] : dp3d(ie,np1,i,j,kp)[ks];
        p_mid = p_int + dp/2;
        p_int += dp;
        p(i,j,kp)[ks] = p_mid;
      }
      // Fill any pack padding with a finite value.
      for (int k = CTI::num_phys_lev; k < CTI::num_lev_pack*CTI::packn; ++k)
        p(i,j,k / CTI::packn)[k % CTI::packn] = p_mid;
    };
    CTI::loop_ij(kv, scan);
    kv.team_barrier();
    sphere_ops.laplace_simple(kv, p, Homme::subview(lap_p, ie));
  };
  Kokkos::fence();
  Kokkos::parallel_for(cti.m_tp_ne, f);
}

void ComposeTransportImpl::advance_hypervis_scalar (const int np1, const Real dt_q) {
  const auto dt = dt_q / m_data.hv_subcycle_q;
  const auto hv_q = m_data.hv_q;
  const auto nu_q = m_data.nu_q;
  const auto Qtens = m_tracers.qtens_biharmonic;
  const auto Q = m_tracers.Q;
  const auto spheremp = m_geometry.m_spheremp;
  const auto tu_ne_hv_q = m_tu_ne_hv_q;
  const auto sphere_ops = m_sphere_ops;
  const bool hv_q_correction = m_data.hv_q_correction > 0;
  if (hv_q_correction) calc_hv_q_lap_p(*this, np1);
  for (int it = 0; it < m_data.hv_subcycle_q; ++it) {
    { // Qtens = Q
      const auto f = KOKKOS_LAMBDA (const int idx) {
        int ie, q, i, j, lev;
        idx_ie_q_ij_nlev<num_lev_pack>(hv_q, idx, ie, q, i, j, lev);
        Qtens(ie,q,i,j,lev) = Q(ie,q,i,j,lev);
      };
      Kokkos::fence();
      launch_ie_q_ij_nlev<num_lev_pack>(hv_q, f);
    }
    // biharmonic_wk_scalar
    const auto laplace_simple_Qtens = [&] () {
      const auto f = KOKKOS_LAMBDA (const MT& team) {
        KernelVariables kv(team, hv_q, tu_ne_hv_q);
        const auto Qtens_ie = Homme::subview(Qtens, kv.ie, kv.iq);
        sphere_ops.laplace_simple(kv, Qtens_ie, Qtens_ie);
      };
      Kokkos::fence();
      Kokkos::parallel_for(m_tp_ne_hv_q, f);
    };
    laplace_simple_Qtens();
    if (hv_q_correction) {
      // Approximate the Laplacian on pressure surfaces:
      //   lap_p(Q) ~ lap_eta(Q) - dQ/dp lap_eta(p),
      // with the local, current dQ/dp smoothly capped at hv_q_thresh. Q is
      // still unmodified here, so it supplies dQ/dp; Qtens holds lap_eta(Q).
      const auto lap_p = m_data.hv_q_lap_p;
      const auto dp3d = m_state.m_dp3d;
      const auto divdp = m_derived.m_divdp;
      const bool use_divdp = m_data.independent_time_steps;
      const Real thresh = m_data.hv_q_thresh;
      const auto f = KOKKOS_LAMBDA (const int idx) {
        const int ie = idx / (hv_q*np*np);
        const int q = (idx / (np*np)) % hv_q;
        const int i = (idx / np) % np;
        const int j = idx % np;
        const auto Qv = [&] (const int k) -> Real {
          return Q(ie,q,i,j,k / packn)[k % packn];
        };
        for (int k = 0; k < num_phys_lev; ++k) {
          const int kp = k / packn, ks = k % packn;
          const Real dp = use_divdp ? divdp(ie,i,j,kp)[ks] : dp3d(ie,np1,i,j,kp)[ks];
          // Q at the layer interfaces, one-sided at the top and bottom.
          const Real q_top = k == 0 ? Qv(0) : (Qv(k-1) + Qv(k))/2;
          const Real q_bot = k == num_phys_lev-1 ? Qv(k) : (Qv(k) + Qv(k+1))/2;
          Real dqdp = (q_bot - q_top)/dp;
          dqdp = dqdp/(1 + (dqdp < 0 ? -dqdp : dqdp)/thresh);
          Qtens(ie,q,i,j,kp)[ks] -= dqdp*lap_p(ie,i,j,kp)[ks];
        }
      };
      Kokkos::fence();
      Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, m_data.nelemd*hv_q*np*np), f);
    }
    m_hv_dss_be[0]->exchange(m_geometry.m_rspheremp);
    if (m_data.hv_scaling == 0) {
      Kokkos::fence();
      laplace_simple_Qtens();
    } else {
      const auto tensorvisc = m_geometry.m_tensorvisc;
      const auto f = KOKKOS_LAMBDA (const MT& team) {
        KernelVariables kv(team, hv_q, tu_ne_hv_q);
        const auto Qtens_ie = Homme::subview(Qtens, kv.ie, kv.iq);
        sphere_ops.laplace_tensor(kv, Homme::subview(tensorvisc, kv.ie),
                                  Qtens_ie, Qtens_ie);
      };
      Kokkos::fence();
      Kokkos::parallel_for(m_tp_ne_hv_q, f);
    }
    { // Compute Q = Q spheremp - dt nu_q Qtens. N.B. spheremp is already in
      // Qtens from divergence_sphere_wk.
      const auto f = KOKKOS_LAMBDA (const int idx) {
        int ie, q, i, j, lev;
        idx_ie_q_ij_nlev<num_lev_pack>(hv_q, idx, ie, q, i, j, lev);
        Q(ie,q,i,j,lev) = (Q(ie,q,i,j,lev) * spheremp(ie,i,j)
                           - dt * nu_q * Qtens(ie,q,i,j,lev));
      };
      Kokkos::fence();
      launch_ie_q_ij_nlev<num_lev_pack>(hv_q, f);
    }
    // Halo exchange Q and apply rspheremp.
    Kokkos::fence();
    m_hv_dss_be[1]->exchange(m_geometry.m_rspheremp);
  }
}

} // namespace Homme

#endif // HOMME_ENABLE_COMPOSE

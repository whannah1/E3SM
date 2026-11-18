#include "catch2/catch.hpp"

#include "share/core/eamxx_types.hpp"
#include "physics/zm/zm_functions.hpp"
#include "physics/zm/tests/infra/zm_test_data.hpp"

#include "zm_unit_tests_common.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace scream {
namespace zm {
namespace unit_test {

/*
 * Tests for the optional plume freezing (use_plume_freezing), which adds the
 * latent heat of fusion of freezing condensate to the ZM updraft MSE budget and
 * removes it from the downstream snow production and detrained ice heating.
 * These are not baseline tests: the default (disabled) path is covered by the
 * existing *_bfb baselines, while the tests here exercise the enabled path.
 */
template <typename D>
struct UnitWrap::UnitTest<D>::TestZmPlumeFreezing : public UnitWrap::UnitTest<D>::Base {

  using ZmRuntimeOpt = typename Functions::ZmRuntimeOpt;
  using ZMC          = typename Functions::ZMC;
  using PC           = typename Functions::PC;

  // tolerance for host recomputation of device results
  static Real rtol() { return ekat::is_single_precision<Real>::value ? 1e-4 : 1e-10; }

  //----------------------------------------------------------------------------
  // ice fraction ramp: end points, mid point, monotone and bounded
  void run_ice_frac()
  {
    const Real tol = std::numeric_limits<Real>::epsilon() * 10;
    const Real t1 = ZMC::plume_frz_tk1, t2 = ZMC::plume_frz_tk2;
    REQUIRE(ZmRuntimeOpt::calc_plume_ice_frac(t1 + 10) == 0);
    REQUIRE(ZmRuntimeOpt::calc_plume_ice_frac(t1)      == 0);
    REQUIRE(ZmRuntimeOpt::calc_plume_ice_frac(t2)      == 1);
    REQUIRE(ZmRuntimeOpt::calc_plume_ice_frac(t2 - 10) == 1);
    REQUIRE(ZmRuntimeOpt::calc_plume_ice_frac(Real(0.5)*(t1+t2)) == Approx(0.5).margin(tol));
    Real prev = 0;
    for (Int n = 0; n <= 100; ++n) {
      const Real f = ZmRuntimeOpt::calc_plume_ice_frac(t1 + 5 - Real(n)*(t1 - t2 + 10)/100);
      REQUIRE(f >= 0);
      REQUIRE(f <= 1);
      REQUIRE(f >= prev);
      prev = f;
    }
  } // run_ice_frac

  //----------------------------------------------------------------------------
  // Fill d with a deterministic, conditionally unstable tropical sounding so
  // that zm_cloud_properties produces an active updraft that rises through the
  // mixed phase range. Uniform layers of thickness dz from the surface to z_top,
  // a 6.5 K/km lapse rate up to an isothermal stratosphere, and relative
  // humidity decreasing from 90% near the surface to 30% aloft. Columns differ
  // in surface temperature and land fraction. k=0 is the model top.
  static void set_convecting_sounding(ZmCloudPropertiesData& d)
  {
    const Real grav  = PC::gravit.value, cpair = PC::Cpair.value, rdair = PC::Rair.value;
    const Real z_top = 20000, dz = z_top / d.pver;
    for (Int i = 0; i < d.pcols; ++i) {
      const Real t_sfc = 300 + Real(0.5)*i;
      d.landfrac[i] = d.pcols > 1 ? Real(i) / (d.pcols - 1) : Real(0);
      d.tpert_g[i]  = 0.5;
      for (Int k = 0; k < d.pverp; ++k) {
        d.z_int[i*d.pverp + k] = (d.pverp - 1 - k) * dz;
      }
      // integrate the hypsometric equation upward from the surface
      Real p_below = 1000; // [mb]
      Real z_below = 0;
      for (Int k = d.pver - 1; k >= 0; --k) {
        const Int  n  = i*d.pver + k;
        const Real z  = Real(0.5) * (d.z_int[i*d.pverp + k] + d.z_int[i*d.pverp + k + 1]);
        const Real t  = std::max(t_sfc - Real(6.5e-3)*z, t_sfc - Real(6.5e-3)*15000);
        const Real p  = p_below * std::exp(-grav * (z - z_below) / (rdair * t));
        const Real rh = z < 2000 ? Real(0.9) : (z < 10000 ? Real(0.9) - Real(0.6)*(z - 2000)/8000 : Real(0.3));
        // saturation specific humidity from Bolton (1980)
        const Real es = Real(6.112) * std::exp(Real(17.67)*(t - Real(273.15)) / (t - Real(29.65)));
        const Real qs = Real(0.622) * es / std::max(p - Real(0.378)*es, es);
        d.z_mid[n] = z;
        d.t_mid[n] = t;
        d.p_mid[n] = p;
        d.q_mid[n] = rh * qs;
        d.s_mid[n] = t + grav / cpair * z;
        p_below = p;
        z_below = z;
      }
      // interface dry static energy: mean of the adjacent layers
      for (Int k = 0; k < d.pver; ++k) {
        const Int n = i*d.pver + k;
        d.s_int[n] = k == 0 ? d.s_mid[n] : Real(0.5) * (d.s_mid[n-1] + d.s_mid[n]);
      }
      // updraft base in the second layer above the surface, launch level near 13 km
      d.jb[i]  = d.pver - 2;
      d.lel[i] = Int(d.pver * (z_top - 13000) / z_top);
    }
  }

  //----------------------------------------------------------------------------
  // zm_cloud_properties with plume freezing enabled
  void run_cloud_properties()
  {
    auto engine = Base::get_engine();

    const Real latvap = PC::LatVap.value, latice = PC::LatIce.value;
    const Real cpair  = PC::Cpair.value,  grav   = PC::gravit.value;
    const Real lat_ratio = latice / latvap;

    // The deterministic sounding must freeze condensate, the random data only
    // has to pass the budget checks where it happens to convect
    struct DataSet {
      ZmCloudPropertiesData d;
      bool require_active;
    };
    std::vector<DataSet> base_data = {
      //                     pcols, ncol, pver, pverp, msg, limcnv
      {ZmCloudPropertiesData(    4,    4,   40,    41,   2,   3), true},
      {ZmCloudPropertiesData(    4,    4,   80,    81,   2,   3), true},
      {ZmCloudPropertiesData(    4,    4,   72,    73,  10,  40), false},
      {ZmCloudPropertiesData(    4,    4,  128,   129,  30,  60), false},
    };
    for (auto& ds : base_data) {
      if (ds.require_active) {
        set_convecting_sounding(ds.d);
      } else {
        ds.d.randomize(engine);
      }
    }

    for (const auto& ds : base_data) {
      const auto& d0 = ds.d;
      const Int pcols = d0.pcols, pver = d0.pver, pverp = d0.pverp;

      // reference run with the default (legacy) interface
      ZmCloudPropertiesData d_off(d0);
      zm_cloud_properties(d_off);

      // the extended interface with freezing disabled must reproduce it exactly
      {
        ZmCloudPropertiesData d_dis(d0);
        ZmPlumeFrzData f_dis;
        f_dis.use_plume_freezing = false;
        zm_cloud_properties(d_dis, &f_dis);
        for (Int i = 0; i < pcols; ++i) {
          REQUIRE(d_dis.cldtop_klev[i] == d_off.cldtop_klev[i]);
          REQUIRE(d_dis.jlcl[i]        == d_off.jlcl[i]);
        }
        for (Int n = 0; n < d_off.total(d_off.ql); ++n) {
          REQUIRE(d_dis.mflx_up[n] == d_off.mflx_up[n]);
          REQUIRE(d_dis.s_upd[n]   == d_off.s_upd[n]);
          REQUIRE(d_dis.q_upd[n]   == d_off.q_upd[n]);
          REQUIRE(d_dis.cu[n]      == d_off.cu[n]);
          REQUIRE(d_dis.ql[n]      == d_off.ql[n]);
          REQUIRE(d_dis.rprd[n]    == d_off.rprd[n]);
          REQUIRE(d_dis.evp[n]     == d_off.evp[n]);
        }
      }

      // enabled
      ZmCloudPropertiesData d(d0);
      ZmPlumeFrzData f;
      zm_cloud_properties(d, &f);
      REQUIRE(static_cast<Int>(f.frz.size()) == pcols*pver);

      Int  n_frz = 0, n_target = 0, n_diff = 0;
      Real tot_frz = 0;
      for (Int i = 0; i < pcols; ++i) {
        const Int jt = d.cldtop_klev[i];
        const Int jb = d0.jb[i];
        if (d.jlcl[i] != d_off.jlcl[i] || jt != d_off.cldtop_klev[i]) ++n_diff;
        const Real c0 = ZMC::c0 * (1 - d.landfrac[i]) + ZMC::c0 * d.landfrac[i];
        for (Int k = 0; k < pver; ++k) {
          const Int  n  = i*pver + k;
          const Real dz = d.z_int[i*pverp + k] - d.z_int[i*pverp + k + 1];
          if (d.s_upd[n] != d_off.s_upd[n] || d.cu[n] != d_off.cu[n]) ++n_diff;

          // sign and bounds
          REQUIRE(f.frz[n]  >= 0);
          REQUIRE(f.qi[n]   >= 0);
          REQUIRE(f.sprd[n] >= 0);
          REQUIRE(d.cu[n]   >= 0);
          REQUIRE(f.qi[n] <= d.ql[n] * (1 + rtol()) + std::numeric_limits<Real>::min());

          // freezing only inside the updraft
          if (k < jt || k >= jb) {
            REQUIRE(f.frz[n] == 0);
            REQUIRE(f.qi[n]  == 0);
            continue;
          }
          if (f.frz[n] > 0) ++n_frz;
          tot_frz += dz * f.frz[n];
          if (k + 1 >= pver) continue;

          // zm_cloud_properties returns rprd net of downdraft evaporation,
          // so add evp back to recover the gross autoconversion rate
          const Real rprd_auto = d.rprd[n] + d.evp[n];
          REQUIRE(f.sprd[n] <= rprd_auto * (1 + rtol()) + std::numeric_limits<Real>::min());
          REQUIRE(std::abs(f.sprd[n] - c0*d.mflx_up[n]*f.qi[n]) <= rtol() * (std::abs(f.sprd[n]) + 1e-30));

          // total condensate budget:
          // flux in + condensation = flux out + detrainment + rain production
          {
            const Real t_in  = d.mflx_up[n+1]*d.ql[n+1];
            const Real t_cu  = dz*d.cu[n];
            const Real t_out = d.mflx_up[n]*d.ql[n];
            const Real t_det = dz*d.detr_up[n]*d.ql[n+1];
            const Real t_rp  = dz*rprd_auto;
            const Real scale = std::abs(t_in) + std::abs(t_cu) + std::abs(t_out)
                             + std::abs(t_det) + std::abs(t_rp) + dz*std::abs(d.evp[n]);
            REQUIRE(std::abs((t_in + t_cu) - (t_out + t_det + t_rp)) <= rtol() * scale);
          }

          // ice budget:
          // flux in + freezing = flux out + detrainment + snow production
          {
            const Real i_in  = d.mflx_up[n+1]*f.qi[n+1];
            const Real i_frz = dz*f.frz[n];
            const Real i_out = d.mflx_up[n]*f.qi[n];
            const Real i_det = dz*d.detr_up[n]*f.qi[n+1];
            const Real i_sp  = dz*f.sprd[n];
            const Real scale = std::abs(i_in) + std::abs(i_frz) + std::abs(i_out)
                             + std::abs(i_det) + std::abs(i_sp);
            REQUIRE(std::abs((i_in + i_frz) - (i_out + i_det + i_sp)) <= rtol() * scale);
          }

          // latent heating: condensation plus fusion must equal the heating
          // implied by the updraft DSE budget (the cu definition without freezing)
          if (k > jt && d.mflx_up[n] > 0) {
            const Real lat_dse = ((d.mflx_up[n]*d.s_upd[n] - d.mflx_up[n+1]*d.s_upd[n+1])/dz
                                  - (d.entr_up[n] - d.detr_up[n])*d.s_mid[n]) / (latvap/cpair);
            const Real lat_tot = d.cu[n] + lat_ratio*f.frz[n];
            const Real scale   = std::abs(d.mflx_up[n]*d.s_upd[n]/dz) / (latvap/cpair);
            REQUIRE(std::abs(lat_tot - std::max(Real(0), lat_dse)) <= rtol() * scale);
          }

          // where the freezing is not limited, the condensate at the top of the
          // layer reaches the target ice fraction of the updraft temperature
          if (f.frz[n] > 0 && d.cu[n] > 0 && d.mflx_up[n] > 0) {
            const Real t_upd = d.s_upd[n] - grav/cpair * d.z_int[i*pverp + k];
            const Real f_ice = ZmRuntimeOpt::calc_plume_ice_frac(t_upd);
            REQUIRE(std::abs(f.qi[n] - f_ice*d.ql[n]) <= rtol() * d.ql[n]);
            ++n_target;
          }
        }
      }

      if (ds.require_active) {
        // fail rather than pass silently if the enabled path was not exercised
        REQUIRE(n_frz > 0);
        REQUIRE(n_target > 0);
        REQUIRE(tot_frz > 0);
        // and freezing must change the plume
        REQUIRE(n_diff > 0);
      }
    }
  } // run_cloud_properties

  //----------------------------------------------------------------------------
  // zm_calc_output_tend: fusion heating added to dsdt, plume ice detrainment
  void run_calc_output_tend()
  {
    auto engine = Base::get_engine();
    std::uniform_real_distribution<Real> frz_dist(0, 1e-5), qfrac_dist(0, 1);

    std::vector<ZmCalcOutputTendData> base_data = {
      //                   pcols, ncol, pver, pverp, msg
      ZmCalcOutputTendData(    4,    4,   72,    73,  10),
      ZmCalcOutputTendData(    4,    4,  128,   129,  30),
    };
    for (auto& d0 : base_data) {
      d0.randomize(engine);
      const Int pcols = d0.pcols, pver = d0.pver;

      // plume freezing inputs: zero at and below cloud base, like zm_cloud_properties
      ZmPlumeFrzData f;
      f.frz.assign(pcols*pver, 0);
      f.qi.assign(pcols*pver, 0);
      for (Int i = 0; i < pcols; ++i) {
        for (Int k = 0; k < d0.mx[i]; ++k) {
          const Int n = i*pver + k;
          f.frz[n] = frz_dist(engine);
          f.qi[n]  = qfrac_dist(engine) * d0.ql[n];
        }
      }

      ZmCalcOutputTendData d_off(d0), d_on(d0);
      zm_calc_output_tend(d_off);
      zm_calc_output_tend(d_on, &f);
      REQUIRE(static_cast<Int>(f.di.size()) == pcols*pver);

      Int ktm = pver;
      for (Int i = 0; i < pcols; ++i) ktm = std::min(ktm, d0.cldtop_klev[i]);

      const Real fus = PC::LatIce.value / PC::Cpair.value;
      for (Int i = 0; i < pcols; ++i) {
        for (Int k = 0; k < pver; ++k) {
          const Int n = i*pver + k;
          // moisture and liquid detrainment are not affected
          REQUIRE(d_on.dqdt[n] == d_off.dqdt[n]);
          REQUIRE(d_on.dl[n]   == d_off.dl[n]);
          if (k >= ktm && k < pver-1) {
            REQUIRE(f.di[n] == d0.detr_up[n]*f.qi[n+1]);
            if (k < d0.mx[i]) {
              const Real expect = fus*f.frz[n];
              REQUIRE(std::abs((d_on.dsdt[n] - d_off.dsdt[n]) - expect)
                      <= rtol() * (std::abs(d_off.dsdt[n]) + std::abs(expect)));
            }
          } else {
            REQUIRE(f.di[n] == 0);
          }
          if (k >= d0.mx[i]) REQUIRE(d_on.dsdt[n] == d_off.dsdt[n]);
        }
      }
    }
  } // run_calc_output_tend

  //----------------------------------------------------------------------------
  // zm_conv_evap: plume ice is not heated again at snow production
  void run_conv_evap()
  {
    auto engine = Base::get_engine();
    std::uniform_real_distribution<Real> frac_dist(0, 1);

    std::vector<ZmConvEvapData> base_data = {
      //             pcols, ncol, pver, pverp, time_step
      ZmConvEvapData(    4,    4,   72,    73, 2.0),
      ZmConvEvapData(    4,    4,  128,   129, 4.0),
    };
    for (auto& d0 : base_data) {
      d0.randomize(engine);
      const Int pcols = d0.pcols, pver = d0.pver, pverp = d0.pverp;

      ZmPlumeFrzData f;
      f.sprd.assign(pcols*pver, 0);
      for (Int n = 0; n < pcols*pver; ++n) f.sprd[n] = frac_dist(engine) * d0.prdprec[n];

      ZmConvEvapData d_off(d0), d_on(d0);
      zm_conv_evap(d_off);
      zm_conv_evap(d_on, &f);

      // precipitation, snow and evaporation are unchanged
      for (Int i = 0; i < pcols; ++i) {
        REQUIRE(d_on.prec[i] == d_off.prec[i]);
        REQUIRE(d_on.snow[i] == d_off.snow[i]);
      }
      for (Int n = 0; n < pcols*pverp; ++n) {
        REQUIRE(d_on.flxprec[n] == d_off.flxprec[n]);
        REQUIRE(d_on.flxsnow[n] == d_off.flxsnow[n]);
      }
      // only the snow production heating changes, by the plume ice fusion heat
      const Real lf = PC::LatIce.value;
      for (Int n = 0; n < pcols*pver; ++n) {
        REQUIRE(d_on.tend_q[n]          == d_off.tend_q[n]);
        REQUIRE(d_on.ntprprd[n]         == d_off.ntprprd[n]);
        REQUIRE(d_on.ntsnprd[n]         == d_off.ntsnprd[n]);
        REQUIRE(d_on.tend_s_snwevmlt[n] == d_off.tend_s_snwevmlt[n]);
        const Real expect = lf*f.sprd[n];
        REQUIRE(std::abs((d_off.tend_s[n] - d_on.tend_s[n]) - expect)
                <= rtol() * (std::abs(d_off.tend_s[n]) + expect));
        REQUIRE(std::abs((d_off.tend_s_snwprd[n] - d_on.tend_s_snwprd[n]) - expect)
                <= rtol() * (std::abs(d_off.tend_s_snwprd[n]) + expect));
      }
    }
  } // run_conv_evap

};

} // namespace unit_test
} // namespace zm
} // namespace scream

namespace {

TEST_CASE("zm_plume_freezing_ice_frac", "[zm]")
{
  using TestStruct = scream::zm::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestZmPlumeFreezing;

  TestStruct t;
  t.run_ice_frac();
}

TEST_CASE("zm_cloud_properties_plume_freezing", "[zm]")
{
  using TestStruct = scream::zm::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestZmPlumeFreezing;

  TestStruct t;
  t.run_cloud_properties();
}

TEST_CASE("zm_calc_output_tend_plume_freezing", "[zm]")
{
  using TestStruct = scream::zm::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestZmPlumeFreezing;

  TestStruct t;
  t.run_calc_output_tend();
}

TEST_CASE("zm_conv_evap_plume_freezing", "[zm]")
{
  using TestStruct = scream::zm::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestZmPlumeFreezing;

  TestStruct t;
  t.run_conv_evap();
}

} // empty namespace

# Input reference

Input files are INI style: `[section]` headers, `key = value`, `#` comments. Every key can be overridden
on the command line as `section.key=value` (no spaces), which is how the study tools build their cases.
Unused keys are reported at the end of a run (typos do not pass silently). Tables (histories) are written
`time, value; time, value; ...` and are piecewise linear, constant outside their range; their breakpoints
are hit exactly by the time stepping.

## [geometry]
| key | default | meaning |
|---|---|---|
| `radius` | 4.67e-3 | pellet radius [m] (paper: diameter 9.34 mm) |
| `height` | 14e-3 | pellet height [m] |
| `n_pellets` | 1 | pellets stacked along z (2 for sympathetic-cracking studies) |

## [mesh]
| key | default | meaning |
|---|---|---|
| `nr` | 12 | radial hex elements (innermost ring = wedges at the axis) |
| `ntheta` | 32 | circumferential elements (multiple of `cracks.radial_planes`) |
| `nz` | 24 | axial elements **per pellet** |
| `radial_grading` | 1 | >1 clusters elements toward the outer surface |

## [cracks] - potential crack planes (cohesive surfaces)
| key | default | meaning |
|---|---|---|
| `radial_planes` | 8 | 0 (none), >= 2 equally spaced half-planes through the axis, or `all` (every element boundary: crack angle free) |
| `theta_offset_deg` | 0 | rotation of the radial planes, one value per pellet (snapped to element boundaries) |
| `n_slabs` | - | number of slabs per pellet -> `n_slabs - 1` equally spaced transverse planes |
| `transverse_z` | - | explicit transverse plane positions as fractions of H (overrides `n_slabs`) |
| `circumferential_radii` | - | circumferential crack surfaces as fractions of R (snapped to node rings) |

Every element face shared by two different "regions" (pellet, slab, sector, ring) becomes a 4+4-node
zero-thickness interface; nodes are duplicated once per region, so crossing planes and the axis junction
need no special treatment.

## [material]
| key | default | meaning |
|---|---|---|
| `youngs_modulus` | 219e9 | Pa (Olander 1976) |
| `poisson_ratio` | 0.345 | |
| `thermal_expansion` | 10e-6 | 1/K |
| `reference_temperature` | 293.15 | strain-free temperature [K] |
| `density` | 10421.5 | kg/m3 (95 % TD) |
| `conductivity_scale` | 1 | multiplier on k(T, Bu) |
| `constant_conductivity` | - | if > 0: constant k [W/m/K] (verification) |
| `constant_specific_heat` | - | if > 0: constant cp [J/kg/K] |
| `burnup_degradation` | true | Lucuta et al. (1996) FD, FP, FR factors on Fink (2000) k(T) |
| `swelling`, `gaseous_swelling` | true | MATPRO solid and gaseous fission-product swelling |

## [strength], [toughness] - Weibull-distributed cohesive properties
| key | default | meaning |
|---|---|---|
| `distribution` | weibull | `weibull` or `constant` |
| `mean` | 130e6 Pa / 4.0 J/m2 | mean of the distribution (the Weibull scale follows from it) |
| `weibull_modulus` | 8 | m |
| `threshold` | 0 | location parameter (3-parameter Weibull) |

## [weibull] - spatial structure of the scatter
| key | default | meaning |
|---|---|---|
| `correlation_length` | 0.5e-3 | lc [m] of a Gaussian random field mapped to the Weibull distributions; mesh-independent. 0 = independent value per interface element |
| `toughness_strength_correlation` | 1 | rho between the strength and toughness fields (1 = same field, 0 = independent) |
| `reference_area` | 0 | for lc = 0: weakest-link scaling sigma0 (A_ref/A_el)^(1/m); 0 = off |
| `points_per_length` | 4 | random-field grid points per lc |
| `seed` | 1 | realisation |
| `flaws` | - | `x,y,z,radius,strength_factor[,toughness_factor]; ...` local weak (or strong) zones |

G_c is raised to 1.05 sigma_t^2/(2K) where a draw would leave no softening branch (count in the manifest).

## [cohesive]
| key | default | meaning |
|---|---|---|
| `stiffness` | 1e16 | initial (penalty) stiffness K [N/m3] (paper value) |
| `shear_stiffness_ratio` | 1 | K_s / K |
| `softening` | linear | `linear` (paper, delta_f = 2 G_c/sigma_t) or `exponential` |
| `mode_mixity_beta` | 0 | effective separation sqrt(<dn>^2 + beta^2 ds^2); 0 = mode I only (paper) |
| `viscosity` | 1e-4 | viscous regularisation mu [s] (Abaqus convention; paper 1e-4) |
| `crack_face_friction` | 0 | Coulomb coefficient on closed, damaged faces |
| `residual_stiffness` | 1e-6 | matrix-only stiffness fraction on failed/open faces (no force) |
| `cracked_damage` | 0.99 | damage above which a point counts as cracked |

## [interface] - pellet-pellet end faces (n_pellets > 1)
| key | default | meaning |
|---|---|---|
| `type` | contact | `contact` (frictional, unilateral) or `cohesive` (bonded) |
| `friction` | 0.5 | Coulomb coefficient, or `stick` |
| `stiffness` | = cohesive | normal penalty [N/m3] |
| `tangential_stiffness_ratio` | 1 | |
| `contact_fraction` | 1 | nominal contact-area fraction: contact only for r >= R sqrt(1 - f) |
| `dish_radius`, `dish_depth` | -, 0.3e-3 | explicit dish (initial gap inside dish_radius) |
| `chamfer_width`, `chamfer_depth` | 0, 0.1e-3 | outer chamfer gap |

## [thermal]
| key | default | meaning |
|---|---|---|
| `convection_coefficient` | 5000 | W/m2/K on the outer surface (paper) |
| `coolant_temperature` | 500 | table or constant [K] |
| `fixed_surface_temperature` | false | Dirichlet T = coolant instead of convection |
| `initial_temperature` | coolant(0) | K |
| `crack_emissivity` | 0.8 | radiation across cracks (view factor 1) |
| `crack_roughness` | 1e-6 | m, added to the opening in k_He/(d + r) |
| `bonded_conductance` | 1e7 | W/m2/K across intact interfaces |
| `pellet_contact_conductance` | 5e4 | solid conductance at the pellet contact |

## [loading]
| key | default | meaning |
|---|---|---|
| `power_history_W_per_cm` | 0 | linear heat rate table [s, W/cm], uniform volumetric heating |

## [boundary]
| key | default | meaning |
|---|---|---|
| `bottom` | roller | `roller` (u_z = 0, frictionless) or `platen` (rigid frictional contact at z = 0) |
| `platen_friction`, `platen_stiffness` | 0, = interface | |
| `top` | free | `free`, `pressure` (table `top_pressure` [Pa] or constant `pressure`), `displacement` (table `top_displacement` [m], rigid frictionless platen) |
| `pin_rigid_body` | true | three statically determinate in-plane pins (reported as `pin_reaction_N`) |

## [time]
| key | default | meaning |
|---|---|---|
| `end`, `dt_initial`, `dt_min`, `dt_max` | 200, 1, 1e-9, 1e7 | s |
| `grow`, `cut` | 1.5, 0.25 | step growth after easy steps / cut after a failure |
| `max_power_change` | 0.05 | max change of power per step (fraction of the max) |
| `max_damage_increment` | 0.5 | reject a step if the viscous damage jumps more than this anywhere |
| `softening_dt_factor` | 2 | dt <= factor x viscosity while cracks grow unstably |
| `initiation_target` | 1.02 | failure-index target used to approach the next crack initiation |
| `max_steps` | 1e6 | |

## [solver]
| key | default | meaning |
|---|---|---|
| `newton_max_iterations` | 40 | modified Newton (secant interface matrix) |
| `newton_tolerance` | 5e-4 | ||R|| / ||F_thermal|| (paper used 5e-3 and 5e-4) |
| `displacement_tolerance` | 1e-6 | secondary criterion |
| `line_search` | true | backtracking |
| `preconditioner` | amg | `amg` (smoothed aggregation) or `jacobi` (block Jacobi) |
| `amg_rebuild_iterations` | 40 | rebuild the hierarchy when PCG needs more iterations |
| `linear_tolerance`, `linear_max_iterations` | 1e-6, 20000 | PCG |
| `stabilization` | 1e-9 | matrix-only diagonal shift (fraction of mean diagonal) |
| `thermal_max_iterations`, `thermal_tolerance` | 30, 1e-3 K | Picard iterations for k(T) |
| `coupling_iterations` | 1 | thermal -> mechanical passes per step |
| `thermal_skip_dt` | 1e-2 | s; steps shorter than this keep the temperature field (crack bursts) |

## [output]
| key | default | meaning |
|---|---|---|
| `directory` | out | run directory |
| `times` | - | output times [s] |
| `every_steps` | 0 | also every N steps |
| `on_crack_area_growth` | 0 | also whenever the crack area grew by this much [m2] |
| `vtu`, `binary` | true, true | write VTU files (base64 binary) |
| `max_vtu` | 400 | |
| `angular_bins` | 72 | bins of the angular damage profiles |
| `labels` | - | `key=value;key=value` copied into manifest.json |

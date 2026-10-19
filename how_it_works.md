# How it works - three_body_project

A private explainer for the owner. Written for a reader who knows physics but
not simulation-based inference (SBI). This file is documentation only:
nothing in code/ or the notebook reads it, imports it, or otherwise depends
on it. Deleting this file changes nothing about how the project runs.

## 1. What the project does, and why a wide posterior can be the right answer

Three point masses move under Newtonian gravity in a 2D plane. Given the
initial positions and velocities, the future is fully determined, but the
system is chaotic: two initial conditions that differ by an immeasurably
small amount produce completely different trajectories after enough time.

The task is the inverse problem: given a short, noisy observation of the
trajectory, infer a posterior distribution over the initial conditions. This
project trains a neural network (BayesFlow, a flow matching posterior
estimator) to do this instantly for any new observation, instead of running
an expensive inference algorithm from scratch each time.

The chaos matters directly to how the results should be read. If the
observation starts long after the true initial time, the system has had a
long time to mix: many different initial conditions, indistinguishable at
the noise level of the observation, could have produced a very similar
trajectory. A correctly working inference method must then report a wide,
sometimes multi-lobed posterior, because that genuinely is the honest state
of knowledge. A narrow, confident posterior in that regime would not be a
better answer; it would be a wrong one, papering over real ambiguity the
data cannot resolve. Section 5 (chaos_summary and posterior_example) and
section 6 below show this directly: contraction (how much the posterior
narrows relative to the prior) is expected and observed to fall as the
observation starts later, while calibration (whether the posterior's stated
uncertainty is honest) stays good throughout. Fading contraction with intact
calibration is the project's headline result, not a defect to fix.

## 2. Every file in code/, and the run order

Each script does one clear job and reads/writes plain files, so any stage
can be rerun on its own as long as the files it depends on already exist.
Run order, empty repo to final figures:

1. simulator.py - no project inputs. Defines the physics (fixed masses in
   ratio 1:2:3, softened gravity, the DOP853 integrator) and, run directly,
   produces a sanity check: results/example_orbits.png and
   results/energy_check.txt. Every other script imports get_masses, EPSILON,
   integrate_system and total_energy from here rather than redefining the
   physics.

2. lyapunov.py - imports simulator.py's physics and prior.py's initial
   condition sampler. No file inputs. Measures how fast nearby trajectories
   diverge (the Lyapunov exponent) over a batch of prior-like draws, and
   writes results/lyapunov.txt (lambda_median, T, K, obs_times, and the
   individual measured rates) plus results/divergence.png. This sets the
   natural timescale everything downstream is built around: T = 3 /
   lambda_median is the length of the observation window, and obs_times are
   K = 10 times spaced through it.

3. prior.py - imports simulator.py's physics. Reads results/lyapunov.txt
   (via read_lyapunov_file, so T, K, obs_times and lambda_median are never
   retyped by hand). Defines the prior sampler (sample_initial_condition) and
   the accept/reject logic (evaluate_initial_condition), used by every script
   downstream that needs a candidate simulation. Not normally run directly
   for output, though it has its own small __main__ smoke check.

4. make_dataset.py - imports simulator.py and prior.py. Reads
   results/lyapunov.txt. Runs prior.py's sampler and rejection check across
   many parallel worker processes, and writes data/train.npz,
   data/validation.npz, data/test.npz and data/dataset_info.txt.

5. train.py - imports simulator.py indirectly through the BayesFlow model
   definition, and device.py directly. Reads data/train.npz and
   data/validation.npz. Builds and trains the BayesFlow workflow, and writes
   models/trained_model.keras, models/training_history.png,
   models/training_info.txt and models/training_log.txt.

6. diagnostics.py - imports device.py. Reads models/trained_model.keras and
   data/test.npz (and results/lyapunov.txt, for lambda_median). Draws
   posterior samples for the whole test set and writes
   results/calibration_ecdf.png, results/recovery.png,
   results/contraction.png and results/chaos_summary.png.

7. device.py - no project inputs or outputs of its own. One function,
   get_device(), used by every script that touches the GPU (train.py,
   diagnostics.py, and the notebook), so the device choice is visible and
   consistent everywhere.

results_notebook.ipynb runs last, after step 6. It reads only
models/trained_model.keras and data/test.npz (plus code/device.py,
code/simulator.py and code/diagnostics.py as helper functions, none of the
other code/ files, and none of results/, data/dataset_info.txt or
docs/), and writes results/posterior_example.png and
results/posterior_predictive.png. It trains nothing.

## 3. Data files, field by field

### data/train.npz, data/validation.npz, data/test.npz

Same schema in all three; N is the number of accepted simulations, K = 10.

- theta, shape (N, 8): the state at t = 0 for bodies 1 and 2 -
  x1, y1, x2, y2, vx1, vy1, vx2, vy2. This is what the network infers. Body
  3 is not stored because it is fixed by the center-of-mass constraints:
  r3 = -(m1 r1 + m2 r2) / m3, v3 = -(m1 v1 + m2 v2) / m3.
- t0, shape (N,): the observation window's start time, drawn uniformly from
  [0, 6 / lambda_median] independently per simulation. A known network
  input, not something being inferred (see section 4).
- obs_noisy, shape (N, K, 12): the noisy observed state (all three bodies,
  positions and velocities) at the K observation times. The only trajectory
  data the network ever sees.
- traj_clean, shape (N, K, 12): the same, without noise. Diagnostics only,
  never a network input.
- obs_times, shape (K,): the K observation times relative to t0 (so the
  absolute time of observation j in simulation i is t0[i] + obs_times[j]).
  Identical across every simulation.
- seed, shape (N,): the random seed that generated that simulation. Every
  simulation is exactly reproducible from its seed alone.
- energy_drift, energy0, angmom0, virial0, min_dist, t_min_dist, walltime,
  shape (N,) each: quality and audit metadata measured or derived per
  simulation (maximum relative energy drift over the run, total energy and
  angular momentum and virial ratio at t = 0, closest approach distance and
  when it happened, integration wall time). Never network inputs; they exist
  so any oddity in the results can be traced back to specific properties of
  the underlying simulation.

### data/dataset_info.txt

Plain key-value text. Split sizes and total rejections
(n_train/n_validation/n_test, n_rejected_total, and the count by reason:
ejection, collision, integrator failure, energy drift), the acceptance
thresholds used (escape_radius, collision_radius, the energy drift
threshold is fixed in prior.py and not repeated here), the physical
constants (epsilon, sigma_x, sigma_v), the timescale (T, K, lambda_median,
t0_max, and window_design, a text label of the observation scheme),
integrator settings, and the base_seed and chunk bookkeeping used to make
generation resumable.

### models/trained_model.keras

The full BayesFlow workflow (adapter, summary network, inference network,
weights) saved with its own save method. Bayesflow 2.0.12 exactly; saved
models are not guaranteed to load across versions.

### models/training_info.txt

Plain key-value text: device, how many epochs actually ran, batch size,
training set size, optimizer and learning rate schedule settings, the
summary and inference network choices, early stopping's patience and
whether and when it triggered, wall time, final train and validation loss,
best validation loss, and which epoch is saved (saved_model_is_best_val
idation_epoch and best_epoch - the checkpoint mechanism, described in
section 6, guarantees the saved file is always the best epoch, not
whichever epoch training happened to stop on).

### models/training_log.txt, models/training_history.png

training_log.txt is one line per epoch (epoch, train loss, val loss) written
live as training runs. training_history.png plots the same two curves.

## 4. How the model works, conceptually

**Prior.** Initial conditions are sampled in the center-of-mass frame:
positions for each body uniform in a disc, then recentered and rescaled so
the mean pairwise separation is 1; velocities normal with a scale tuned from
the potential energy so the virial ratio is of order one, then recentered so
total momentum is zero. A candidate is rejected if any pairwise distance
exceeds an escape radius or falls below a collision radius, if the
integrator fails, or if the energy drift over the run exceeds 1e-8.

**Simulator.** Newtonian gravity with a small softening length (epsilon =
0.02) so close encounters do not blow up numerically, integrated with
scipy's DOP853 at tight tolerances (rtol 1e-10, atol 1e-12).

**Noise.** Independent Gaussian noise is added to every recorded position
(sigma_x = 0.01) and velocity (sigma_v = 0.01) component, at every
observation time, before the network ever sees the trajectory.

**The t0 window design, and why it exists.** Early in the project, every
simulation was observed over the same fixed window [0, T]. That made it
impossible to honestly ask "how does inference quality change with how late
the observation starts," since every training example started at the same
time; a later version of that question had to truncate the fixed window
after the fact, which fed the network something it had never been trained
on. The fix was to make the window start itself, t0, a random variable
drawn per simulation from [0, 6 / lambda_median], with the system still
integrated from the true t = 0 but only observed from t0 onward for a
duration T. t0 is told to the network as a known condition (the observer
always knows what time they started watching), not something being
inferred. This makes "how does the posterior change with observation
lateness" a question the trained model can answer honestly, in
distribution, by simply looking at test cases with different t0, instead of
an out-of-distribution guess. Section 6 has the full before/after story.

**Summary network.** A time-aware network (BayesFlow's TimeSeriesNetwork, a
recurrent/convolutional architecture) reads the K noisy observed states and
compresses them into a fixed-size summary vector. This is what lets the
model handle a trajectory instead of a single point.

**Flow matching inference network.** Given the summary vector and t0 as
conditions, this network defines a continuous transformation from a simple
base distribution to the posterior over the 8 theta parameters. Sampling
means drawing from the base distribution and running the transformation
forward; it naturally represents wide, skewed or multi-lobed posteriors,
which is why it was chosen over a simpler coupling-flow architecture for a
chaotic problem where multimodality is expected.

**What "amortized" means here.** The network is trained once, offline, on
many simulated (observation, true initial condition) pairs. After training,
producing a posterior for a brand new observation is a single fast forward
pass and some sampling, not a new optimization or MCMC run. The upfront
training cost is "amortized" over every future observation the model is
ever asked about.

## 5. How to read each figure in results/

### results/chaos_summary.png

The headline figure. The test set is binned by observation lateness, t0 in
units of 1/lambda (six bins of width 1 over [0, 6]). For each bin, mean
posterior contraction and mean calibration error across all 8 parameters
are plotted against bin center. GOOD: contraction decreases smoothly as
lateness grows (posteriors get honestly wider, not narrower, as chaos
erodes identifiability) while calibration error stays low and roughly flat
across all bins (the width is honest at every lateness, not just on
average). BAD: contraction staying flat near 1 regardless of lateness
(the model claims equal confidence no matter how little information is
really there, which is the overconfidence failure mode this project hit and
fixed - see section 6), or calibration error rising sharply at high
lateness (the model is honestly wide but wrong about how wide).

### results/recovery.png

One panel per parameter, posterior mean plotted against the true value
across the test set, with credible interval error bars, and the
correlation r printed on each panel. GOOD: points close to the diagonal
line with a high r; error bars that are neither absurdly tight (given the
correlation is not near 1) nor so wide the plot is uninformative. BAD:
points scattered with no relationship to the diagonal (r near 0, the model
learned nothing), or very tight error bars sitting far from the diagonal
(confidently wrong).

### results/contraction.png

A bar chart, one bar per parameter, of posterior contraction (1 = posterior
collapses to a point relative to the prior, 0 = posterior as wide as the
prior) pooled across the whole test set (all values of t0 mixed together).
GOOD: bars at a moderate height, reflecting that the test set mixes easy
(small t0) and hard (large t0) cases, so a value well short of 1 is
expected and correct, not a shortfall. BAD: every bar pinned near 1
regardless of the underlying mix of easy and hard cases (the overconfidence
pattern again), or bars near 0 (the model is not using the observation at
all).

### results/calibration_ecdf.png

One panel per parameter: simulation-based calibration (SBC), the rank of
the true value within the posterior samples, plotted as an empirical CDF
difference from the ideal uniform rank distribution, with a 95 percent
confidence band. This is computed once over the whole test set (all t0
values pooled). GOOD: the curve stays inside the gray band for its full
length. BAD: the curve wanders outside the band, especially in a smooth,
consistent shape across many parameters (a systematic bias or width
problem, not noise). See section 7: this project's aggregate version of
this plot still sits outside the band even though the per-bin calibration
error in chaos_summary.png is low, an open, documented tension, not
something papered over.

### results/posterior_example.png (written by the notebook)

A corner plot (histograms on the diagonal, pairwise scatter below it) of
posterior samples for one easy (small t0) and one hard (large t0) test
case, with the true value marked. GOOD: the small-t0 case's cloud is tight
and roughly single-peaked around the true value; the large-t0 case's cloud
is visibly wider and may show separate blobs or curved, non-Gaussian
shapes. Both are good outcomes for their respective case. BAD: the large-t0
case looking just as tight as the small-t0 case (overconfidence, again), or
the small-t0 case's cloud missing the true value entirely (the model failed
even the easy case).

### results/posterior_predictive.png (written by the notebook)

For the same two example cases, several posterior samples are re-simulated
forward and overlaid on the real noisy observation: a faint dashed segment
for the time before the observation window starts (unconstrained context
only), a solid segment for the observation window itself (the part actually
being checked), and the real noisy data points on top. GOOD: the solid
segments cluster near the actual observed points in both panels. This shows
the posterior samples are dynamically consistent with producing something
like the real observation; it does NOT by itself prove those samples are
close to the true initial condition, since a chaotic system can make
different initial conditions look similar at the noise level - that
stronger claim is what calibration_ecdf.png and chaos_summary.png are for.
BAD: the solid segments passing nowhere near the actual observed points
(the posterior samples are not dynamically plausible at all, a much more
basic problem than a calibration shortfall).

## 6. The two-design story

**Design 1: fixed window [0, T].** Every simulation was observed over the
same window. The trained model reached a very low training loss and, on the
full window, showed posterior contraction near 1 for every parameter with
tight credible intervals - looking, at a glance, excellent. But the
aggregate SBC rank ECDF was clearly out of its confidence band for every
parameter in the same systematic shape, the signature of an overconfident
(too narrow) posterior rather than a genuinely well-calibrated one. Testing
"how does inference change with observation lateness" under this design
required truncating the fixed window at inference time to fewer than the
trained K = 10 steps - an input shape the network had never seen in
training. That sweep showed contraction staying pinned near 1 regardless of
how little was truncated to, while calibration error swung erratically
between about 0.07 and 0.44 with no clear trend: neither the "contraction
fades" nor the "calibration survives" pattern chaos theory predicts, most
plausibly because the network was answering an out-of-distribution question
it was never trained to handle, not because that is the true behavior of
the underlying inference problem.

**Design 2: the t0 window redesign.** Section 4 describes the mechanism.
With t0 a real, in-distribution training input, the chaos summary sweep
became a simple bin-by-t0 measurement on the real test set instead of an
input-shape hack. The first retrain under this design (still saving the
final training epoch, see below) already showed the qualitatively correct
pattern: contraction fell from about 0.88 at low lateness to about 0.51 at
high lateness while calibration error stayed roughly flat, around 0.09 to
0.11.

**The best-validation checkpoint fix.** Separately, it was noticed that
training always saved whichever epoch training happened to stop on, not the
epoch with the best validation loss - and the validation loss curve
visibly bottomed out partway through training and drifted back up
afterward (a normal overfitting pattern), so the saved model was sometimes
measurably worse than the best point already reached. train.py was changed
to checkpoint the model to disk only when validation loss improves (so the
file on disk is always the best-epoch weights, with no separate save step
needed after training) and to stop early after 20 epochs with no
improvement. Retraining with this fix and rerunning diagnostics on the
correctly-saved model cut the chaos_summary calibration error from roughly
0.09-0.11 down to roughly 0.018-0.033 (about 0.025 on average) at every
lateness bin, while contraction fell further and more steeply, from about
0.76 at low lateness to about 0.26 at high lateness. Saving the actual best
model, not whichever epoch training stopped on, was worth roughly a factor
of four in calibration error - a much bigger effect than any of the
architecture or dataset changes tried up to that point.

## 7. Known limitations, stated honestly

**The aggregate SBC ECDF still sits outside the 95 percent band.** Even on
the best-validation model, results/calibration_ecdf.png (computed by
pooling every test simulation across all values of t0 into one SBC test)
shows several parameters' rank ECDF curves outside the confidence band,
though the deviation is smaller than under either earlier design. The
per-bin calibration error in results/chaos_summary.png, by contrast, is low
and stable in every lateness bin individually. The working hypothesis,
not yet confirmed, is that pooling smears the rank distribution because the
posterior's width itself changes a great deal across the pooled range of
t0 (contraction from about 0.76 down to about 0.26), even when each narrow
slice of t0 is individually well calibrated. This is logged as open, not
resolved.

**Posterior draws can fall outside the effective prior support.** The
1e-8 energy drift acceptance threshold used to build the training data was
designed for prior draws, which prior.py's rejection sampling filters for
escape and collision before they ever reach the dataset. Posterior samples
are not filtered that way: the trained network is free to place some
probability mass on initial conditions that lead to a near-collision, and
when such a sample is re-simulated the integrator genuinely strains through
the close encounter, occasionally exceeding that same 1e-8 threshold. A
scratch measurement found this in roughly 1.7 percent of re-simulated
posterior draws, and it was observed for real in the notebook's own run (1
of 30 draws in one example case). This is treated as expected and
informative, not an error: the notebook reports the drift and the count of
exceedances rather than treating it as a failure, and an occasional small
exceedance is itself a hint about where the posterior places implausible
mass, worth knowing rather than hiding.

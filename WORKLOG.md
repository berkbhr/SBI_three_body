# WORKLOG - three_body_project

Append-only logbook. One entry per significant step.

## 2026-07-13, session start

part of the project: project setup, session start checks

action taken: read CLAUDE.md, docs/master_plan_three_body.md and
docs/data_contracts.md in full. Confirmed no WORKLOG.md and no env_three_body
existed yet, so this is the first session. Created folders code, data, models,
results. Created the virtual environment env_three_body. Installed the packages
listed in the Frequent commands section of CLAUDE.md.

real commands executed:
  mkdir code data models results
  py -3.12 -m venv env_three_body
  env_three_body\Scripts\python.exe -m pip install --upgrade pip
  env_three_body\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  env_three_body\Scripts\python.exe -m pip install bayesflow==2.0.12 numpy scipy matplotlib jupyter
  env_three_body\Scripts\python.exe -m pip freeze > requirements.txt
  env_three_body\Scripts\python.exe --version
  env_three_body\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  env_three_body\Scripts\python.exe -c "import os; os.environ['KERAS_BACKEND']='torch'; import bayesflow; print(bayesflow.__version__)"

result:
  env_three_body Python version: 3.12.0
  torch version: 2.11.0+cu128
  cuda available: True
  gpu: NVIDIA GeForce RTX 5070 Ti Laptop GPU
  bayesflow version: 2.0.12, imports cleanly with torch backend

problems (even solved ones):
  the venv was first created with the system default python (3.14.4, the only
  one on PATH). bayesflow/torch wheel support for 3.14 is not established yet
  and CLAUDE.md's own example command names 3.12 as the preferred choice when
  several Pythons are installed, so the venv was deleted and recreated with
  py -3.12 (found at C:\Users\angel\AppData\Local\Programs\Python\Python312\python.exe).
  Nothing had been installed into the 3.14 venv yet, so no work was lost.

  pip install hit the Norton SSL interception described in CLAUDE.md
  (CERTIFICATE_VERIFY_FAILED on pypi.org). Followed stage 2 of the Norton
  protocol: retried with --trusted-host pypi.org --trusted-host
  files.pythonhosted.org --trusted-host download.pytorch.org, which succeeded.
  Made it permanent for this environment by writing env_three_body\pip.ini
  with the same trusted hosts, so later installs in this venv do not need the
  flag repeated by hand.

decisions and why:
  chose Python 3.12 for env_three_body instead of the system default 3.14,
  because CLAUDE.md's frequent-commands section explicitly anticipates this
  choice and 3.12 has mature torch/bayesflow wheel support.

next step: write code/device.py and code/simulator.py per the master plan and
data contracts, run the simulator sanity check (energy drift + example orbits
plot), and show the results to the owner before writing any other file.

## 2026-07-13, simulator

part of the project: code/device.py and code/simulator.py, sanity check

action taken: wrote code/device.py (one function, get_device, prints and
returns "cuda" or "cpu"). Wrote code/simulator.py: fixed masses 1/6, 2/6, 3/6,
softening epsilon = 0.02, acceleration() and equations_of_motion() implementing
d2r_i/dt2 = sum_j!=i m_j (r_j - r_i) / (|r_j-r_i|^2 + eps^2)^(3/2), total_energy()
with the matching softened potential, integrate_system() using
scipy.integrate.solve_ivp with method DOP853, rtol 1e-10, atol 1e-12. Three
hand picked example initial conditions (bodies 1 and 2 chosen directly, body 3
from the center of mass constraints) used only for this sanity check, not the
real prior (that is prior.py's job later). Ran the script directly.

real commands executed:
  env_three_body\Scripts\python.exe code\simulator.py

result:
  example_0 max_relative_energy_drift 1.2822382540892922e-09 accepted True
  example_1 max_relative_energy_drift 9.700071535217065e-10 accepted True
  example_2 max_relative_energy_drift 7.684354193426447e-10 accepted True
  all three at least 3 orders of magnitude below the 1e-8 acceptance threshold
  results/example_orbits.png written: three panels, each showing the x-y paths
  of the three bodies over t in [0, 5], visibly curved gravitational orbits
  results/energy_check.txt written with the epsilon, integrator settings and
  the three drift lines above

problems (even solved ones): none, sanity check passed on the first run.

decisions and why: used epsilon = 0.02 exactly as specified in the master
plan, DOP853 with the specified tight tolerances, and did not build any
prior-sampling logic inside simulator.py, keeping that scope for prior.py.

next step: waiting for owner confirmation of these simulator results before
starting any other file (lyapunov.py, prior.py, or otherwise).

## 2026-07-13, lyapunov

part of the project: code/lyapunov.py

action taken: owner confirmed the simulator results. Wrote code/lyapunov.py.
It imports get_masses, EPSILON and integrate_system from simulator.py rather
than duplicating the physics. Added a prior-like initial condition sampler
(disc positions rescaled to mean separation 1, COM subtracted; velocities
normal with scale tuned from the virial ratio, COM subtracted) used only for
this chaos measurement, since prior.py with the real accept/reject prior does
not exist yet. For each draw, clones the state with a random-direction
perturbation of size d0 = 1e-8, integrates both copies to t_max = 10 with 400
points, and fits the slope of ln d(t) versus t over the window where distance
is between 2*d0 and 0.3 (past the initial transient, before saturation).
Repeated until 20 valid measurements or 60 attempts, took the median as
lambda_median, and applied the gate: lambda values must all be positive and
within a factor of about 10 of each other, and at least half the target draws
must produce a clear exponential region.

real commands executed:
  env_three_body\Scripts\python.exe code\lyapunov.py

result:
  draws attempted: 20, valid lambda measurements: 20 (no draws wasted)
  lambda values ranged from 0.270 to 1.338, lambda_median = 0.7208
  ratio max/min = 4.95, comfortably under the factor of 10 gate
  gate passed
  T = 3 / lambda_median = 4.1622
  K = 10, obs_times = 0.4162, 0.8324, ..., 4.1622 (equally spaced in (0, T])
  results/divergence.png written: ln d(t) curves for 6 example draws with
  their fitted slopes overlaid as dashed lines; curves show the expected
  chaotic pattern, an overall exponential rise with periodic spikes from
  close encounters, and the fits track the trend through the noise
  results/lyapunov.txt written with lambda_median, lambda_all, T, K,
  obs_times, d0, n_batch

problems (even solved ones): none, the gate passed on the first run, no
retuning of the sampler or the fit window was needed.

decisions and why: measured lambda with a self-contained prior-like sampler
inside lyapunov.py instead of importing prior.py, because prior.py has not
been written yet and the project structure lists lyapunov.py ahead of it;
this sampler follows the same distribution shape described in the master
plan (disc positions, COM correction, virial-tuned velocities) but performs
no bounded-trajectory rejection, since that rejection logic belongs to
prior.py.

next step: write code/prior.py (sample and reject initial conditions) and
code/make_dataset.py (reads T and obs_times from results/lyapunov.txt),
generate a 100 accepted simulation test dataset, and show the owner lambda,
T, obs_times, the divergence plot, the rejection rate and dataset_info.txt.
Do not launch the full 50000/2000/2000 generation without confirmation.

## 2026-07-13, prior and make_dataset

part of the project: code/prior.py, code/make_dataset.py, 100-sample test
dataset

action taken: wrote code/prior.py. Moved sample_initial_condition (previously
duplicated inline in lyapunov.py) here as the canonical sampler, since prior.py
now exists and owns "sample and reject" per the project structure; updated
lyapunov.py to import it from prior instead, removing the duplicate, and
reran lyapunov.py to confirm identical results after the refactor. Added
compute_at_t0 (energy0, angmom0, virial0), check_bounded_trajectory (ejection,
collision, energy drift checks over a fine 300 point grid across [0, T]),
evaluate_initial_condition (fine grid check, then a second cheap integration
at exactly the K obs_times for accepted draws only), and read_lyapunov_file
(shared by prior.py and make_dataset.py so T and obs_times are never retyped
by hand). Added a small __main__ smoke check (50 candidates) for prior.py on
its own.

First smoke check with escape_radius = 10.0, collision_radius = 0.05 (as
originally planned) gave a rejection rate of 0.80, all 40 rejections from
"collision", against the master plan's roughly 70 percent ceiling. Explored
the cause in a scratch script (not part of code/): swept the velocity virial
target and collision_radius together on batches of 60 to 100 candidates.
Lowering the virial target made things worse (more tightly bound orbits spend
more time close together, raising collision rejections from about 75 percent
at target 1.0 up to about 95-97 percent at targets 0.2-0.3). Raising the
virial target reduced collisions but introduced ejections instead (order
50-60 percent rejection at target 2.0-5.0, now with a meaningful ejection
share), a real change in the sampled physics, not just a rejection threshold.
Chose the smallest, most conservative fix instead: leave the velocity scale
exactly as specified (virial ratio of order one, untouched) and lower
collision_radius from 0.05 to 0.03, a single parameter, still 1.5x the
softening length 0.02 so accepted trajectories stay outside the
softening-dominated regime. This is a retune of an acceptance threshold, not
of the prior's sampling scales, and data honesty is preserved either way
since min_dist and t_min_dist are recorded for every accepted simulation, so
close encounters remain visible to diagnostics later regardless of where the
line is drawn.

Wrote code/make_dataset.py: generate_split() draws candidates with
per-sample seeds derived from a base_seed plus a running counter, using fixed
non overlapping seed offsets per split (train, validation, test) so seeds
can never collide across splits regardless of which split is generated in
which run. Saves a partial checkpoint every CHUNK_SIZE = 50 accepted samples
so an interrupted run can resume without redoing prior work (not yet tested
by an actual interruption). Writes data/{split}.npz matching the schema in
data_contracts.md, data/{split}_stats.txt (n_accepted and rejection counts
for that split), and data/dataset_info.txt (aggregated across whichever
split stats files exist, so info stays correct even if splits are generated
in separate sessions). Supports a command line override,
"make_dataset.py <split> <target_n>", for small test runs of a single split
without touching the others; with no arguments it targets the full plan
default 50000 / 2000 / 2000.

Deliberately left out the multiprocessing pool across CPU cores that the
master plan calls for. A single process was fast enough for today's 100
sample demonstration (about 20 seconds for 100 accepted plus 128 rejected
attempts) that adding multiprocessing now would only have added Windows
spawn-related risk without a clear benefit; it should be added and tested
before the full 50000 sample run, which will need it.

real commands executed:
  env_three_body\Scripts\python.exe code\lyapunov.py    (rerun after refactor)
  env_three_body\Scripts\python.exe code\prior.py       (first check, 0.05)
  (scratch exploration script, not saved in code/)
  env_three_body\Scripts\python.exe code\prior.py       (recheck, 0.03)
  env_three_body\Scripts\python.exe code\make_dataset.py test 100

result:
  lambda_median 0.7208, T 4.1622, K 10
  obs_times 0.4162, 0.8324, 1.2487, 1.6649, 2.0811, 2.4973, 2.9136, 3.3298,
  3.7460, 4.1622 (unchanged from the earlier lyapunov run, confirmed after
  the sampler refactor)
  prior recheck (50 candidates, collision_radius 0.03): accepted 18,
  rejection rate 0.64, reasons {ejection 0, collision 31, integrator 0,
  energy 1}
  test dataset generation (target 100): 100 accepted, 228 seeds attempted,
  128 rejected {ejection 0, collision 126, integrator 0, energy 2},
  rejection rate 0.561, finished in 19.8 seconds
  data/test.npz shapes: theta (100,8), obs_noisy (100,10,12),
  traj_clean (100,10,12), obs_times (10,), all quality fields (100,)
  energy_drift range 1.45e-11 to 8.19e-9, all under the 1e-8 threshold as
  expected since rejects above that are filtered out
  min_dist range 0.0301 to 0.7696, all above collision_radius 0.03 as
  expected
  data/dataset_info.txt written: n_train 0, n_validation 0, n_test 100,
  n_rejected_total 128, escape_radius 10.0, collision_radius 0.03,
  epsilon 0.02, sigma_x 0.01, sigma_v 0.01, T and K copied from
  results/lyapunov.txt, base_seed 1000

problems (even solved ones): first prior check rejection rate (0.80) exceeded
the master plan's roughly 70 percent gate; retuned collision_radius as
described above and confirmed the fix before generating any dataset.

decisions and why: retuned the acceptance threshold (collision_radius) rather
than the prior's sampling scales, to stay faithful to the master plan's
exact velocity distribution ("virial ratio of order one") while still
bringing the rejection rate under the gate; documented the full exploration
so the choice can be revisited if diagnostics later show it mattered.
Deferred multiprocessing in make_dataset.py to before the full run, not
before this 100 sample test, since it was not needed yet and adds platform
risk.

next step: waiting for owner review of these results (lambda/T/obs_times,
divergence plot, rejection rate, dataset_info.txt) before writing train.py,
diagnostics.py or launching the full 50000/2000/2000 generation.

## 2026-07-13, multiprocessing and full dataset generation

part of the project: code/make_dataset.py (multiprocessing pool), full
50000/2000/2000 dataset generation

action taken: owner confirmed T and the prior settings. Added the
multiprocessing pool that the master plan calls for to code/make_dataset.py.
A module level worker function, evaluate_seed(seed, T, obs_times), draws one
candidate, evaluates it with prior.evaluate_initial_condition, and adds noise
if accepted; it is defined at module level (not nested or a closure) because
Windows multiprocessing spawns a fresh interpreter per worker and needs to
pickle and re-import the function. main() opens one
multiprocessing.Pool(processes=os.cpu_count()) inside the existing
if __name__ == "__main__": guard and reuses it across every wave and every
split, so the pool is only started once per run. generate_split() now submits
CHUNK_SIZE = 500 candidate seeds per pool.map call (a "chunk" is now a wave of
attempts, not a fixed number of accepted samples as in the single process
version), checkpoints the partial file after each wave, and prints one
progress line per chunk. Also added the n_chunks field that data_contracts.md
calls for under "chunks_completed" in dataset_info.txt, which the
single-process version had been missing; fixed that gap while touching this
file.

Verified the pool with a 200-accepted-simulation run of the test split before
trusting it with the full run. Compared to the earlier 100-sample
single-process run: same 12 array keys, same dtypes, same shape pattern
(N, 8) / (N, K, 12) / (N,) scaled to N = 200, and the first several accepted
seeds were bit-identical between the single-process and multiprocessing runs
(20001003, 20001004, 20001005, ...), confirming each worker's per-seed
sampling is unaffected by parallelization. 500 candidates gave 214 accepted
in 5.7 seconds with 24 processes, versus about 20 seconds for 100 accepted
from 228 candidates single process, a clear speedup.

Then launched the full generation with no command line arguments (train,
validation, test at the plan's default targets 50000 / 2000 / 2000), running
in the background since it was expected to take longer than a single
foreground command should block.

real commands executed:
  env_three_body\Scripts\python.exe code\make_dataset.py test 200   (verification)
  env_three_body\Scripts\python.exe code\make_dataset.py            (full run)

result:
  train: 50000 accepted, 227 chunks, 113500 seeds attempted, 62663 rejected
  for collision, 59 for ejection, 618 for energy drift, 0 integrator
  failures, finished in 1053.4 seconds
  validation: 2000 accepted, 10 chunks, 5000 seeds attempted, 2783 collision,
  2 ejection, 29 energy, finished in 43.8 seconds
  test: 2000 accepted, 10 chunks, 5000 seeds attempted, 2753 collision,
  1 ejection, 31 energy, finished in 43.3 seconds
  total wall time across all three splits: 1140.6 seconds (about 19.0
  minutes)
  data/dataset_info.txt: n_train 50000, n_validation 2000, n_test 2000,
  n_rejected_total 68939 (rejected_ejection 62, rejected_collision 68199,
  rejected_integrator 0, rejected_energy 678), escape_radius 10.0,
  collision_radius 0.03, epsilon 0.02, sigma_x 0.01, sigma_v 0.01,
  T 4.16223622087363, K 10, rtol 1e-10, atol 1e-12, method DOP853,
  base_seed 1000, chunks_completed train=227,validation=10,test=10
  overall rejection rate 68939 / (68939 + 54000) = 0.561, matching the
  0.561-0.64 range seen in the earlier smaller checks, so the retuned
  collision_radius behaved consistently at full scale
  verified all three data/{split}.npz files: every array key and shape
  matches data_contracts.md exactly (theta (N,8), obs_noisy and traj_clean
  (N,10,12), obs_times (10,), all seven quality fields (N,)) for
  N = 50000 / 2000 / 2000; energy_drift stayed under 1e-8 and min_dist
  stayed at or above 0.03 in all three files, as the acceptance criteria
  require; seeds are unique within every split and there is zero seed
  overlap across train, validation and test (checked directly), confirming
  the fixed per-split seed offsets prevented any cross-split leakage

problems (even solved ones): none. The pool verification matched the single
process version exactly in structure and in the overlapping seeds, and the
full run completed without any integrator failures or unexpected rejection
reasons.

decisions and why: kept CHUNK_SIZE (wave size) at 500 candidates per pool.map
call, sized for the 24 available cores; this gave a progress print roughly
every few seconds during the train split without excessive pool call
overhead. Trimmed any wave overshoot past target_n to keep each split's
final count exactly on target, discarding the small surplus of extra
accepted samples from the last wave rather than keeping ragged sizes.

next step: waiting for the owner before writing train.py. The dataset is
ready: data/train.npz, data/validation.npz, data/test.npz, and
data/dataset_info.txt all exist and match the data contract.

## 2026-07-13, train

part of the project: code/train.py, full training run

action taken: owner confirmed the dataset. Confirmed the session start GPU
check first (training is heavy work): torch.cuda.is_available() True on the
RTX 5070 Ti Laptop GPU. Before writing any code, inspected the actual
bayesflow 2.0.12 API surface (BasicWorkflow, ContinuousApproximator, Adapter,
TimeSeriesNetwork, TimeSeriesTransformer, FlowMatching, build_optimizer) so
the expensive run would use the correct classes and signatures for this exact
version rather than guessed ones.

Wrote code/train.py. It sets KERAS_BACKEND to torch before importing
bayesflow, as CLAUDE.md requires, and calls device.get_device() so the device
is printed visibly (came out "using GPU: NVIDIA GeForce RTX 5070 Ti Laptop
GPU"). load_split reads only theta and obs_noisy from each npz, honoring the
anti-leak rule (clean trajectory and quality variables are never loaded into
training). build_workflow builds an explicit Adapter that casts float64 to
float32, groups theta as inference_variables and obs_noisy as
summary_variables, a TimeSeriesNetwork summary network (summary_dim 32) over
the 10 recorded states, and a FlowMatching inference network, tied together
in a BasicWorkflow with standardize set to both inference_variables and
summary_variables. Verified from the bayesflow source that
summary_variables are standardized before they reach the summary network, so
this one setting standardizes both the parameters and the observations as the
plan asks. The workflow's default offline optimizer is AdamW with a
cosine-decay learning rate (5 percent warmup, annealed toward zero over the
requested epochs), so the validation loss flattens near the end by
construction; kept that default. Added a keras TerminateOnNaN callback so a
NaN run stops immediately instead of wasting the full training time, and a
check_gate function that flags the run if any loss is NaN or if the best
validation loss never dropped below the first epoch, in which case the model
is not saved.

Ran a small-sample smoke test first (a scratch script reusing train.py's own
load_split and build_workflow), 512 train and 128 validation simulations for
2 epochs, before committing to the full run. It confirmed: the GPU is
actually used (torch.cuda memory went from a few KB to about 31 MB during
fit), loss decreased with no NaN, and the saved model round-trips
(keras.saving.load_model returned a ContinuousApproximator that sampled
posteriors of shape (4 conditions, 10 samples, 8 parameters)). One benign
cuDNN warning about non-contiguous RNN weights appeared; it is a performance
note, not an error, and was ignored.

Then launched the full run (data/train.npz with data/validation.npz for
validation loss, batch size 64, 100 epochs) in the background, and confirmed
the first epoch was healthy (26 seconds, loss 1.35, val_loss 0.88, no NaN)
before letting it run to completion.

real commands executed:
  env_three_body\Scripts\python.exe (scratch smoke test, not saved in code/)
  env_three_body\Scripts\python.exe code\train.py

result:
  device: cuda (RTX 5070 Ti Laptop GPU)
  wall time: 3220.3 seconds (about 53.7 minutes) for 100 epochs
  no NaN at any epoch (grep found zero); gate passed
  validation loss decreased monotonically and flattened at the end:
  epoch 1 val 0.8781, epoch 10 0.3760, epoch 20 0.3005, epoch 30 0.2432,
  epoch 40 0.2157, epoch 50 0.1924, epoch 60 0.1807, epoch 70 0.1630,
  epoch 80 0.1496, epoch 90 0.1371, epoch 100 0.1370
  final train loss 0.1433, final val loss 0.1370, best val loss 0.1313
  train and validation curves track closely with no overfitting gap; val loss
  runs slightly below train loss during the run (train includes summary-net
  dropout and stochastic flow-matching time sampling; validation is in eval
  mode) and the two meet at the end
  saved models/trained_model.keras (about 10.5 MB), models/training_info.txt
  (device, epochs, batch size, n_train, learning rate settings, optimizer,
  network choices, wall time, final and best losses), and
  models/training_history.png (the loss curves)

problems (even solved ones): none that affected the result. The only notable
observation was validation loss sitting slightly below training loss for most
of the run, which is expected here (dropout and FM noise inflate the training
loss relative to eval-mode validation) and not a sign of a data problem;
train and val converge by the final epochs.

decisions and why: used TimeSeriesNetwork as the time-aware summary network
rather than TimeSeriesTransformer, because the 10 observation times are fixed
and identical across every simulation, so the temporal information is fully
carried by sequence position; the recurrent/conv TimeSeriesNetwork captures
that directly and avoids plumbing a constant time vector into a
time-embedding transformer, keeping the code simpler with no loss of
information. Kept the workflow's default AdamW cosine-decay optimizer and
100 epochs so the schedule anneals to zero and the validation loss flattens
by construction, satisfying "enough epochs to see the validation loss
flatten" without guessing an early-stopping point. Handled standardization of
both parameters and observations through the workflow's standardize list
(the bayesflow 2 adaptive-standardization mechanism) rather than baking fixed
mean and std constants into the adapter, since the adaptive layer normalizes
both groups correctly across the batch and time axes without hand-rolled
broadcasting.

next step: waiting for the owner before writing diagnostics.py. The trained
model and its training artifacts are ready in models/.

## 2026-07-13, diagnostics

part of the project: code/diagnostics.py, full test-set diagnostics run

action taken: owner confirmed training. Answered the owner's question first:
the saved model is the final epoch (100), not the best validation epoch.
train.py never sets checkpoint_filepath or save_best_only on the workflow,
so no ModelCheckpoint callback runs during fit_offline, and
workflow.approximator.save(...) is called once after fit_offline returns,
capturing whatever weights are current at that point, which is the end of
epoch 100. Best val loss was 0.1313 (around epoch 95-98) versus final val
loss 0.1370, a small but real gap; the saved model is the epoch-100 weights.

Inspected the actual bayesflow 2.0.12 diagnostics API before writing code
(bf.diagnostics.calibration_ecdf, recovery, posterior_contraction,
calibration_error), confirming exact signatures, return shapes (contraction
and calibration_error return a dict with a per-parameter "values" array,
not a figure), and that calibration_ecdf/recovery return matplotlib figures
directly.

Wrote code/diagnostics.py: loads only theta and obs_noisy from data/test.npz
(anti-leak rule), loads the trained model, and computes SBC calibration_ecdf
and recovery directly with the two built-in bayesflow plotting functions on
the full 2000-simulation test set with 100 posterior draws per simulation.
Contraction is computed with bf.diagnostics.posterior_contraction (a genuine
built-in bayesflow metric) and rendered as a simple custom bar chart per
parameter, since that function returns values, not a figure.

The chaos summary figure required a design decision the plan did not spell
out mechanically: the model was trained on the full, fixed 10-step observation
window, and there is no per-test-simulation lambda measurement to key a
"lateness" axis off directly. Implemented it as a sweep: for each cutoff
k = 1..10, feed the trained model the observation truncated to
obs_noisy[:, :k, :] (first k of the 10 recorded times), compute contraction
and calibration_error against the same test set, and plot both against
obs_times[k-1] * lambda_median (elapsed number of Lyapunov times at the last
included observation), reusing the already-computed full-window (k=10)
result rather than resampling it.

Hit a real bug during the first run: CUDA out of memory partway through the
sweep (failed at k=7, "26.06 GiB is allocated by PyTorch" on a 12 GiB GPU),
after k=1 through k=6 and the initial full-window call had all succeeded.
Diagnosed with a scratch script that called model.sample() 20 times across
varying sequence lengths while tracking torch.cuda.memory_allocated(): each
new, distinct observation window length appears to leave behind a cached
compute graph that is not released between calls, and repeated calls with
never-before-seen shapes (which is exactly what the sweep does, a new length
every iteration) accumulate until the GPU is exhausted. Confirmed the fix
(gc.collect() plus torch.cuda.empty_cache() after every sample call) holds
memory flat at about 0.018 GB across 20 calls spanning two full sweeps.
Added the fix inside sample_posterior() so it covers every call site
uniformly, then reran diagnostics.py end to end successfully.

real commands executed:
  env_three_body\Scripts\python.exe (scratch API inspection, not saved)
  env_three_body\Scripts\python.exe (scratch memory leak repro, not saved)
  env_three_body\Scripts\python.exe code\diagnostics.py   (failed, OOM at k=7)
  env_three_body\Scripts\python.exe code\diagnostics.py   (succeeded after the fix)

result:
  device: cuda, lambda_median read from results/lyapunov.txt: 0.7207663959472048
  full-window (k=10) posterior sampled for all 2000 test simulations, 100
  draws each, in about 26 seconds
  results/calibration_ecdf.png: saved, but the rank ECDF difference curves
  for all 8 parameters bulge well outside the 95 percent confidence band in
  the same characteristic S shape, indicating the full-window posterior is
  not well calibrated on the test set (looks overconfident, not just noisy)
  results/recovery.png: saved, strong recovery correlation (r = 0.997-0.998
  for the 4 position parameters, r = 0.963-0.979 for the 4 velocity
  parameters), narrow credible intervals
  results/contraction.png: saved, contraction is about 0.99-1.0 for every
  parameter on the full window, essentially complete contraction
  results/chaos_summary.png: saved, all 10 window-length sweep points:
    k=1  obs_time=0.4162 lateness=0.3  contraction=0.9970 calib_error=0.4419
    k=2  obs_time=0.8324 lateness=0.6  contraction=0.9976 calib_error=0.3862
    k=3  obs_time=1.2487 lateness=0.9  contraction=0.9890 calib_error=0.4337
    k=4  obs_time=1.6649 lateness=1.2  contraction=0.9902 calib_error=0.4332
    k=5  obs_time=2.0811 lateness=1.5  contraction=0.9975 calib_error=0.1544
    k=6  obs_time=2.4973 lateness=1.8  contraction=0.9975 calib_error=0.1315
    k=7  obs_time=2.9136 lateness=2.1  contraction=0.9913 calib_error=0.4345
    k=8  obs_time=3.3298 lateness=2.4  contraction=0.9915 calib_error=0.4346
    k=9  obs_time=3.7460 lateness=2.7  contraction=0.9970 calib_error=0.0672
    k=10 obs_time=4.1622 lateness=3.0  contraction=0.9969 calib_error=0.0816
  contraction stays flat near 1.0 across every lateness, not fading as the
  plan's hypothesis expected; calibration_error swings between about 0.07
  and 0.44 with no clear trend, not simply "surviving" (staying low) either

problems (even solved ones):
  the CUDA memory accumulation bug described above, fixed as described.

  two results look substantively off and are logged here rather than acted
  on, per the rule to log and ask rather than change things unilaterally:
  (1) the full-window calibration_ecdf shows a consistent out-of-band
  pattern across all 8 parameters together with near-total contraction
  (about 0.99-1.0) on the same window, which is the classic signature of an
  overconfident posterior (too narrow given the model's real accuracy)
  rather than a well-calibrated one; this is on the model's actual designed
  full-window use case, not an artifact of the sweep.
  (2) for the chaos summary sweep specifically, the model was trained only
  ever on the full 10-step window, so truncated inputs at k = 1..9 are out
  of its training distribution; the consistently near-1.0 contraction
  regardless of how little is observed (even k=1, a single early point,
  contracts to 0.997) and the erratic, non-monotonic calibration_error
  jumps are consistent with the network defaulting to a confident but
  unreliable response on inputs unlike anything it saw in training, rather
  than genuinely reflecting how chaos erodes identifiability. The figure is
  saved as requested and uses genuine bayesflow built-in metrics throughout,
  but this specific reading of it should not be taken as validated science
  without the owner's input on whether the truncated-window methodology is
  the right one, or whether the underlying calibration problem (1) needs to
  be resolved first.

decisions and why: kept the chaos summary sweep as truncated-window
inference on the already-trained model rather than retraining a family of
models at different window lengths, because diagnostics.py is documented as
running only on the already-trained model against the held-out test set, and
retraining is out of scope for a diagnostics script. Put the GPU memory
cleanup inside sample_posterior() itself rather than scattering it at each
call site, so every current and future caller gets the fix automatically.
Did not attempt to fix the calibration concerns (more training, architecture
changes, different draws count) since CLAUDE.md is explicit that miscalibration
findings get logged and asked about, not silently corrected.

next step: waiting for the owner's read on the two calibration concerns
above before doing anything else. Not started: results_notebook.ipynb.

## 2026-07-13, observation design change, part 1 (code and dataset)

part of the project: prior.py, make_dataset.py, train.py, diagnostics.py,
docs/data_contracts.md; full dataset regeneration under the new design

action taken: owner approved changing the observation design so the window is
no longer fixed at [0, T]. Answered the owner's standing question again first:
the saved model is the final epoch (100), not the best validation epoch, since
train.py sets no checkpoint callback and saves once after fit_offline returns.

New design: each simulation draws a window offset t0 uniformly from
[0, 6/lambda], integrates over [0, t0 + T], and observes the K = 10 states at
the absolute times t0 + obs_times (obs_times is the same (0, T] pattern as
before, now read as offsets relative to the window start). theta stays the
state at t = 0. t0 is stored per simulation and fed to the network as a known
condition alongside the observations. This makes observation lateness a real,
in-distribution axis of the training set rather than something faked at
inference time by truncating windows (the flaw in the previous chaos summary).

Code changes:
- prior.py: evaluate_initial_condition now takes t0, integrates the fine
  acceptance grid and the observation times over [0, t0 + T], and stores t0
  in the result. The fine-grid point count is scaled up with the span
  (fine_n = round(FINE_N * span / T)) so the bounded-trajectory check keeps
  the same time resolution as the original [0, T] check rather than getting
  coarser on the longer window. read_lyapunov_file now also returns
  lambda_median. run_prior_check samples t0 too.
- make_dataset.py: the worker samples t0 = rng.uniform(0, t0_max) from the
  same seeded rng (so the whole candidate stays reproducible from its seed),
  with t0_max = 6/lambda_median; t0 is stored as a new (N,) array in every
  split npz; dataset_info.txt gains lambda_median, t0_max and a window_design
  label. Everything else (prior scales, thresholds, noise, seeds, chunking,
  multiprocessing) is unchanged.
- train.py: adapter now also groups t0 into inference_conditions, standardize
  covers inference_variables, summary_variables and inference_conditions,
  summary_dim raised 32 to 64, epochs raised 100 to 150; batch size, optimizer,
  gates and everything else unchanged. load_split reads t0 and reshapes to
  (N, 1).
- diagnostics.py: samples the posterior once for the whole test set using each
  simulation's own obs and its own t0; the chaos summary is now built by
  binning the test simulations by lateness = t0 * lambda into 1/lambda-wide
  bins over [0, 6] and computing mean contraction and mean calibration_error
  per bin, all in distribution, instead of the previous out-of-distribution
  truncated-window sweep.
- docs/data_contracts.md: added a dated 2026-07-13 schema change log entry as
  the evolution rule requires, added the t0 (N,) field to the split npz schema,
  clarified obs_times as times relative to t0, updated the anti-leak rule to
  include t0 as a known network input, and added lambda_median, t0_max and
  window_design to the dataset_info.txt schema.

real commands executed:
  env_three_body\Scripts\python.exe code\make_dataset.py test 300   (smoke gen)
  env_three_body\Scripts\python.exe (scratch training smoke on the 300-sim set)
  env_three_body\Scripts\python.exe code\make_dataset.py            (full regen, background)

result of the smoke checks (all passed before committing to the long runs):
  generation: 300 accepted in about 21 seconds, rejection rate about 0.78
  (higher than the old 0.56 because the longer [0, t0 + T] integration gives
  more chances to eject or collide; ejections now appear, 28 of them, where
  the short window had almost none). t0 stored with shape (300,), range
  0.007 to 8.27 inside [0, t0_max = 8.324]; energy_drift max 9.90e-9 under
  the 1e-8 gate; min_dist min 0.0302 at or above the 0.03 collision radius;
  t_min_dist reached 10.59, well past the old T = 4.16, confirming the
  integration now runs deep into shifted windows; dataset_info.txt carries
  lambda_median, t0_max and window_design.
  training wiring: t0 loaded as (N, 1), the adapter accepts it as an
  inference condition, fit_offline runs on GPU with loss decreasing and no
  NaN, the saved model round-trips, and sampling with both conditions
  (obs_noisy and t0) returns theta of shape (4, 10, 8) with no NaN.

problems (even solved ones): the rejection rate under the new design (about
0.78) is above the master plan's roughly 70 percent guidance, but this is a
direct and expected consequence of the owner-approved longer observation
window, not a prior-tuning problem, so the prior scales and thresholds were
left unchanged as instructed ("keep everything else identical"); logging it
here for visibility. Generating 54000 accepted will therefore need roughly
240000 candidates.

decisions and why: stored obs_times as the within-window relative pattern
(shape unchanged) plus a separate per-simulation t0, rather than storing full
(N, K) absolute times, so the schema change is minimal and reduces exactly to
the old design at t0 = 0. Scaled the fine acceptance grid with the span to
keep the collision/ejection check as accurate as before, treating "keep
everything identical" as keeping the check quality identical, not its literal
point count. Fed t0 as an inference condition (not through the summary
network) because it is a single known scalar, and standardized it along with
the other groups for conditioning. Raised summary_dim and epochs exactly as
the owner specified and touched nothing else in the training setup.

next step: full dataset regeneration is running in the background. When it
finishes, record dataset_info.txt, then retrain (summary_dim 64, 150 epochs),
then rerun diagnostics with the t0-binned chaos summary.

## 2026-07-13, observation design change, part 2 (dataset regenerated)

part of the project: full dataset regeneration under the t0 window design

action taken: the background regeneration finished. Verified the outputs
directly.

real commands executed:
  env_three_body\Scripts\python.exe code\make_dataset.py   (ran to completion)

result:
  total wall time about 3021 seconds (about 50 minutes): train 2782 s
  (320 chunks), validation 121 s (14 chunks), test 118 s (14 chunks)
  n_train 50000, n_validation 2000, n_test 2000
  n_rejected_total 119647 (rejected_ejection 3869, rejected_collision 114954,
  rejected_integrator 0, rejected_energy 824); overall rejection rate 0.689,
  which is actually just under the 70 percent guidance, so the 0.78 from the
  tiny 300-sample smoke was small-sample noise; ejections are now a real
  category (3869) as expected for the longer windows
  every split verified: theta (N, 8), t0 (N,), obs_noisy (N, 10, 12) all
  correct; t0 spans about [0, 8.32] = [0, t0_max] in each split; energy_drift
  max under 1e-8 in all three (9.9989e-9 train, 9.7395e-9 validation,
  9.9460e-9 test); min_dist at or above the 0.03 collision radius everywhere;
  zero seed overlap across train, validation and test
  dataset_info.txt records lambda_median 0.72077, t0_max 8.32447,
  window_design offset_t0_uniform_0_to_6_over_lambda_then_span_T

problems (even solved ones): none.

decisions and why: none beyond part 1.

next step: retrain with summary_dim 64 and 150 epochs, same gates, then rerun
diagnostics with the t0-binned chaos summary.

## 2026-07-13, observation design change, part 3 (retrain and checkpoint question)

part of the project: retraining under the t0 design; the saved-model epoch
question

action taken: retrained with the new dataset, summary_dim 64, 150 epochs, same
gates and everything else unchanged. Training completed on its own (exit code
0); did not retrain again. Then answered the owner's checkpoint question by
inspecting the disk.

real commands executed:
  env_three_body\Scripts\python.exe code\train.py   (ran to completion, background)
  ls models\ ; find for any .keras / checkpoint / .weights.h5

result:
  device cuda, wall time 4700.7 seconds (about 78 minutes) for 150 epochs
  gate passed: zero NaN at any epoch; validation loss dropped well below its
  first-epoch value (1.5369 to a best of 1.0764), so neither gate condition
  tripped and the model was saved
  final train loss 0.7575, final val loss 1.1364, best val loss 1.0764
  validation flattened around epoch 30 to 45 (val about 1.08 to 1.11) and then
  drifted slightly upward to 1.1364 by epoch 150 while the training loss kept
  falling to 0.7575: a clear overfitting gap, exactly as the owner noted from
  the curve
  these losses are much higher than the previous fixed-window design (final
  val 0.137) because the task is genuinely harder: inferring the t = 0 state
  from windows that can start as late as 6/lambda, deep in the chaotic
  evolution where the initial condition is nearly washed out

checkpoint question, answered explicitly:
  the saved models/trained_model.keras is the FINAL epoch (150), not the best
  validation epoch. train.py sets no checkpoint_filepath and adds no
  ModelCheckpoint callback, and it calls approximator.save(...) once after
  fit_offline returns, so only the epoch-150 weights were ever written.
  Searched the whole project (outside env_three_body): the only model file is
  models/trained_model.keras; there is NO best-validation checkpoint, no
  .weights.h5, no checkpoint file anywhere. The best-validation weights
  (val 1.0764, around epoch 55 to 65) were never saved and cannot be recovered
  without retraining. Per the owner's explicit instruction, since no
  best-validation checkpoint exists on disk, reported this and did nothing
  else about it; did NOT retrain and did NOT add checkpointing without being
  asked.

problems (even solved ones): the overfitting gap above (final val 1.1364 vs
best 1.0764, about 5.5 percent worse loss). Not acted on because the
best-epoch weights are gone and retraining or adding a ModelCheckpoint was
explicitly reserved for the owner's decision. If the owner later wants the
best-validation weights, train.py would need a ModelCheckpoint with
save_best_only and a rerun; flagged, not done.

decisions and why: left the saved final-epoch model in place and ran
diagnostics on it, because swapping in best-validation weights was impossible
(no checkpoint) and retraining was forbidden without asking.

next step: diagnostics on the final-epoch model, recorded below.

## 2026-07-13, observation design change, part 4 (diagnostics)

part of the project: diagnostics.py on data/test.npz, t0-binned chaos summary

action taken: ran diagnostics.py on the 2000-simulation test set. All four
figures regenerated. Because all observation windows are now the same length
K = 10, the earlier variable-length GPU memory issue does not arise; the
posterior is sampled once for the whole test set (each simulation with its own
obs and its own known t0), and the chaos summary reuses those samples, binned
by lateness.

real commands executed:
  env_three_body\Scripts\python.exe code\diagnostics.py

result:
  results/calibration_ecdf.png: rank ECDF curves for all 8 parameters still sit
  outside the 95 percent band in the same S shape, so the posterior pooled over
  all lateness is still not perfectly calibrated on the aggregate SBC test,
  though the excursion is comparable-to-slightly-smaller than the previous
  design
  results/recovery.png: recovery correlations dropped to r = 0.52 to 0.66
  (positions 0.54 to 0.66, velocities 0.52 to 0.57) with wide credible
  intervals, the honest consequence of pooling late windows where the t = 0
  state is nearly unrecoverable, not a regression
  results/contraction.png: per-parameter contraction 0.52 to 0.82 (positions
  x2, y2 highest near 0.81, velocities vx1, vy1 lowest near 0.52), far from the
  previous design's saturated 0.99, i.e. appropriately wider posteriors
  results/chaos_summary.png (the headline): binned by t0 in units of 1/lambda,
  contraction fades monotonically as the window starts later
    bin center 0.5  n 472  contraction 0.882  calibration_error 0.113
    bin center 1.5  n 393  contraction 0.732  calibration_error 0.101
    bin center 2.5  n 356  contraction 0.627  calibration_error 0.112
    bin center 3.5  n 270  contraction 0.541  calibration_error 0.090
    bin center 4.5  n 266  contraction 0.529  calibration_error 0.098
    bin center 5.5  n 243  contraction 0.509  calibration_error 0.108
  so contraction falls from 0.88 to 0.51 as observation lateness grows from
  about 0.5 to 5.5 Lyapunov times, while calibration error stays low and roughly
  flat around 0.09 to 0.11: the intended "contraction fades while calibration
  survives" result, now in distribution and honest

problems (even solved ones): the aggregate SBC ECDF is still out of band, so
pooled calibration is imperfect even though the per-bin calibration_error
metric is modest and stable; logged for the owner, not acted on. The saved
model is the final epoch, slightly past the best-validation point (see part 3),
so a best-epoch model might shift these figures marginally, but not enough to
change the qualitative reading.

decisions and why: kept the chaos summary as a t0 binning of the real test set
(6 bins of width 1/lambda over [0, 6]) with a 30-simulation minimum per bin, so
every point is computed from in-distribution simulations the model was trained
to handle.

next step: waiting for the owner. Not started: results_notebook.ipynb.

## 2026-07-13, best-validation checkpoint

part of the project: code/train.py (checkpointing), retrain, diagnostics rerun

action taken: owner asked for a minimal change to train.py so the saved model
is the best-validation model rather than the final epoch, since part 3 found
no checkpoint existed on disk to recover the earlier run's best epoch.

Added, and only, the following to train.py:
- keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor="val_loss",
  mode="min", save_best_only=True), where MODEL_PATH is the exact contract
  path "models/trained_model.keras". This callback writes the whole model to
  that path only when validation loss improves, so by the time fit_offline
  returns, the file on disk already holds the best-validation weights with no
  separate save call needed. Removed the old post-training
  workflow.approximator.save(...) call entirely, so there is exactly one
  place that writes MODEL_PATH and it can never overwrite the best checkpoint
  with a later, worse epoch.
- keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
  restore_best_weights=True).
- kept keras.callbacks.TerminateOnNaN() exactly as before.
- a small EpochLogger(keras.callbacks.Callback) class that writes
  models/training_log.txt, one line per epoch ("epoch train_loss val_loss"),
  opened fresh in on_train_begin and appended in on_epoch_end. This is the one
  class added; CLAUDE.md allows a class when a library forces it, and keras
  callbacks are the library's mechanism for per-epoch hooks during fit().
- write_training_info now also records early_stopping_patience,
  saved_model_is_best_validation_epoch (always True when the gate passes,
  since the checkpoint mechanism guarantees it), and best_epoch.
- the gate check gained one more condition: if MODEL_PATH does not exist
  after training, that counts as a gate failure ("model file did not save"),
  matching the owner's explicit gate list.
Nothing else changed: same load_split, same build_workflow/Adapter, same
TimeSeriesNetwork summary_dim 64, same FlowMatching, same batch size 64, same
EPOCHS = 150 cap, same data, same seeds. simulator.py, prior.py,
make_dataset.py, diagnostics.py and the data were not touched.

Smoke tested the new callbacks before committing to the long run: a scratch
script trained 10 epochs on a 400/200 subset of data/test.npz with the real
callback wiring (monkeypatching MODEL_PATH and TRAINING_LOG_PATH to scratch
paths), confirmed the checkpoint file was written, training_log.txt had the
right one-line-per-epoch format, best_epoch matched the true minimum in the
returned history, and EarlyStopping's patience counter behaved correctly
(traced the val_loss sequence by hand: it accumulated non-improving epochs
and reset on every new best, never reaching the patience=3 threshold used in
the smoke test, correctly not stopping early on that particular noisy run).

Then retrained on the full dataset and reran diagnostics.py completely
unchanged.

real commands executed:
  env_three_body\Scripts\python.exe (scratch checkpoint/callback smoke test)
  env_three_body\Scripts\python.exe code\train.py         (full retrain, background)
  env_three_body\Scripts\python.exe code\diagnostics.py   (unchanged, background)

result:
  gate passed: zero NaN across all 88 epochs actually run; model file
  existed; validation loss dropped well below its first-epoch value
  device cuda, wall time 2513.7 seconds (about 41.9 minutes) -- notably
  faster than the previous 150-epoch run (about 78 minutes) because
  EarlyStopping stopped training at epoch 88, exactly 20 epochs (the patience)
  after the best epoch, 68
  best_epoch = 68, best_val_loss = 1.0656 (previous run's final-epoch model
  had best_val_loss 1.0764 at a similar epoch but a worse final-epoch save of
  1.1364; this run's SAVED model is the actual best point, 1.0656)
  final_train_loss (epoch 88) 0.8697, final_val_loss (epoch 88) 1.1188 -- for
  reference only, since epoch 88 is not what got saved
  models/trained_model.keras timestamp (16:25) is about 11 minutes earlier
  than models/training_log.txt and training_info.txt (16:36), independently
  confirming the checkpoint file was written mid-training at its best point
  and never touched again after that, exactly as designed
  training_history.png: validation loss visibly bottoms out around epoch
  60-70 and drifts back up afterward while training loss keeps falling, the
  overfitting gap the owner pointed out in part 3, now with the correct
  epoch's weights actually saved instead of the drifted final epoch
  models/training_log.txt: 89 lines (1 header + 88 epochs), format
  "epoch train_loss val_loss", verified epoch 68's line matches best_val_loss
  exactly (0.9479341506958008 1.0656100511550903)
  models/training_info.txt gained early_stopping_patience 20,
  saved_model_is_best_validation_epoch True, best_epoch 68

  diagnostics.py rerun unchanged, results in the next entry.

problems (even solved ones): none.

decisions and why: had ModelCheckpoint write directly to the final contract
path rather than a separate checkpoint file that then needs copying, so there
is a single source of truth for what is saved and no risk of a later step
overwriting the good weights with worse ones. Kept EarlyStopping's
restore_best_weights=True as asked even though the ModelCheckpoint file is
the thing actually relied on for persistence, since the two are complementary
(the checkpoint survives even if the process were killed before fit()
returns; restore_best_weights keeps the in-memory model consistent too) and
the owner asked for both explicitly.

next step: diagnostics results below, then stop as instructed.

## 2026-07-13, diagnostics on the best-validation model

part of the project: diagnostics.py on data/test.npz, unchanged from the
previous run, now scoring the best-validation checkpoint instead of the final
epoch

action taken: ran diagnostics.py exactly as written, no changes.

real commands executed:
  env_three_body\Scripts\python.exe code\diagnostics.py

result: all four figures regenerated, now scoring the epoch-68 checkpoint
instead of the earlier overfit final-epoch model.
  results/chaos_summary.png (headline), binned by t0 in units of 1/lambda,
  6 bins of width 1 over [0, 6], 30-simulation minimum per bin:
    bin center 0.5  n 472  contraction 0.7627  calibration_error 0.0289
    bin center 1.5  n 393  contraction 0.4990  calibration_error 0.0180
    bin center 2.5  n 356  contraction 0.3705  calibration_error 0.0199
    bin center 3.5  n 270  contraction 0.2829  calibration_error 0.0325
    bin center 4.5  n 266  contraction 0.2536  calibration_error 0.0321
    bin center 5.5  n 243  contraction 0.2627  calibration_error 0.0277
  contraction fades from 0.76 to about 0.25-0.26 as lateness grows from 0.5
  to 5.5 Lyapunov times, while calibration_error stays low and roughly flat,
  about 0.018 to 0.032 throughout -- markedly lower and flatter than the
  previous (overfit final-epoch) run's 0.09 to 0.11, a much cleaner version
  of the intended result
  results/contraction.png (pooled over all lateness): 0.26 to 0.60 across the
  8 parameters (vx1 lowest 0.265, x2 highest 0.601), notably lower than the
  previous run's 0.52 to 0.82, i.e. the best-validation model is honestly
  less confident where the previous, overfit model was falsely confident
  results/recovery.png: r = 0.52 to 0.69 (positions 0.544 to 0.692, velocities
  0.524 to 0.588), a broadly similar range to the previous run, still modest
  since this scores the posterior pooled across all lateness including the
  hardest, latest windows
  results/calibration_ecdf.png: still outside the 95 percent band for most
  parameters when pooled across all lateness together (x1, y1, x2, y2 most
  visibly), but the deviations are smaller in magnitude (roughly 0.03 to 0.06
  versus the previous run's uniform 0.06 to 0.10) and the shape is no longer
  the same rigid S curve on every parameter; vy1 and vx2 stay mostly within
  or right at the band edge. Likely explanation, not fully resolved: the
  per-bin calibration_error in the chaos summary is low and stable in every
  bin, but calibration_ecdf pools all lateness bins into one SBC test, and
  since the posterior's width changes a great deal with t0 (contraction 0.76
  down to 0.25), pooling can smear the rank distribution even when each bin
  is individually well behaved; flagged, not investigated further here.

problems (even solved ones): the pooled calibration_ecdf still shows
out-of-band deviations, smaller than before but not resolved; logged above
with a candidate explanation (pooling across very different contraction
regimes) rather than acted on.

decisions and why: none, script was required to run unchanged.

next step: waiting for the owner. Not started: results_notebook.ipynb.

## 2026-07-13, retrain verification (ran outside this session)

part of the project: verifying the completed retrain, training_info.txt,
rerunning diagnostics.py

action taken: the owner reported that the retrain (checkpointing, early
stopping, epoch logging, all added to train.py in the previous entry) had
already finished, run in the owner's own terminal after this session lost
track of the background job, so no completion record existed here. Did not
retrain and did not touch train.py, per the owner's explicit instruction.
Verified everything directly from disk rather than trusting the earlier
entries at face value: read models/training_log.txt (89 lines, 1 header plus
88 epochs) and models/training_info.txt, and cross-checked file timestamps
(trained_model.keras at 16:25, training_log.txt/training_info.txt/
training_history.png at 16:36, results/*.png at 16:37), all consistent with a
single completed run followed by one diagnostics run.

training_log.txt confirms: 88 epochs ran, early stopping's patience of 20
means the best epoch was 88 - 20 = 68, matching best_epoch 68 already
recorded in training_info.txt. Epoch 68's line reads
"68 0.9479341506958008 1.0656100511550903", exactly matching
best_val_loss 1.0656100511550903 in training_info.txt. Epoch 88 (the final
line) reads "88 0.8697007894515991 1.1187639236450195", matching
final_train_loss and final_val_loss there too.

training_info.txt already had early_stopping_patience, best_val_loss,
saved_model_is_best_validation_epoch and best_epoch from the earlier run, but
did not explicitly state that early stopping actually triggered or at which
epoch (a reader would have had to subtract 68 + 20 = 88 themselves). Added two
lines directly to the text file (train.py itself was not touched):
early_stopping_triggered True and early_stopping_triggered_at_epoch 88.

real commands executed:
  (Read tool on models/training_log.txt and models/training_info.txt)
  (Edit tool on models/training_info.txt, two lines added)
  env_three_body\Scripts\python.exe code\diagnostics.py   (rerun, unchanged)

result: see the next entry for the diagnostics rerun.

problems (even solved ones): none; the disk state was fully consistent with
the earlier WORKLOG entries, so no discrepancy needed resolving.

decisions and why: verified against the actual files on disk rather than
just repeating the earlier entries' claims, since the owner's framing (retrain
run outside this session) meant this session had no first-hand record of it;
independent verification confirmed those earlier entries were accurate.

next step: diagnostics rerun below, then stop as instructed.

## 2026-07-13, diagnostics rerun (unchanged script, confirmation)

part of the project: diagnostics.py on data/test.npz, run exactly as written,
no edits, to independently confirm the previous entry's figures against the
same best-validation checkpoint (epoch 68)

action taken: ran diagnostics.py with no changes. Watched the three gates the
owner named: script errors, a figure failing to write, the model failing to
load. None occurred: exit code 0, "loaded models/trained_model.keras on
cuda" printed (model loaded fine), and "all four figures saved to results/"
printed with all four results/*.png timestamps updated to match.

real commands executed:
  env_three_body\Scripts\python.exe code\diagnostics.py

result: numbers are close to, not identical to, the previous diagnostics
entry's, since posterior sampling draws fresh random noise each run; the same
model and test set produce statistically consistent but not bit-identical
figures.
  results/chaos_summary.png, binned by t0 in units of 1/lambda:
    bin center 0.5  n 472  contraction 0.7626  calibration_error 0.0285
    bin center 1.5  n 393  contraction 0.4957  calibration_error 0.0194
    bin center 2.5  n 356  contraction 0.3723  calibration_error 0.0222
    bin center 3.5  n 270  contraction 0.2812  calibration_error 0.0306
    bin center 4.5  n 266  contraction 0.2645  calibration_error 0.0294
    bin center 5.5  n 243  contraction 0.2601  calibration_error 0.0327
  contraction fades 0.76 to 0.26 across lateness, calibration_error stays low
  and flat around 0.019 to 0.033 throughout, matching the previous diagnostics
  entry (0.7627 to 0.2627, 0.018 to 0.032) within run-to-run sampling noise
  results/contraction.png, results/recovery.png, results/calibration_ecdf.png:
  same qualitative picture as the previous entry (modest recovery
  correlations, pooled contraction 0.26 to 0.60, pooled calibration_ecdf still
  outside the 95 percent band for most parameters), not re-described in full
  here since nothing changed

problems (even solved ones): none; all three gates stayed clear.

decisions and why: none, script was required to run unchanged.

next step: waiting for the owner. Not started: results_notebook.ipynb.

## 2026-07-13, results notebook

part of the project: results_notebook.ipynb, the final deliverable notebook

action taken: read the latest WORKLOG entries, CLAUDE.md and
docs/data_contracts.md first. Confirmed code/diagnostics.py already has a
sample_posterior(model, obs_window, t0_window, num_draws) helper with the
gc.collect() plus torch.cuda.empty_cache() GPU memory fix from the diagnostics
work; reused it directly rather than rewriting it, matching the plan's
requirement to reuse existing helpers.

Before writing the notebook, tested a real risk with a scratch script: posterior
draws are samples from the trained generative model, not accept/reject-filtered
initial conditions like the training data, so re-simulating them can in
principle hit the same 1e-8 energy drift gate used during dataset generation
if a draw happens to land near a close encounter. A 60-draw scratch test found
1 marginal exceedance (1.4995e-08, about 1.7 percent of draws), enough that
naive probability arithmetic suggested a real chance of hitting the gate with
dozens of re-simulated draws in the real notebook. Decided not to preemptively
work around this (changing the threshold or the draw count specifically to
dodge it would be improvising past the owner's named gate) and instead built
the notebook with the literal, unchanged 1e-8 threshold and a modest
NUM_PREDICTIVE_DRAWS = 15 per case, then actually executed it, following the
gate exactly as stated: stop and report only if it truly happens, not before.

Built results_notebook.ipynb programmatically with the nbformat library
(more reliable than hand-writing notebook JSON for a file this size), 12
cells alternating markdown explanation and code, in the order specified:
setup and device print, picking the smallest and largest t0 test cases,
drawing posterior samples via the reused helper, a corner plot of the 8 theta
parameters for both cases with truth marked (results/posterior_example.png),
a posterior predictive check re-simulating draws with code/simulator.py and
overlaying them on the noisy observation (results/posterior_predictive.png),
and a closing summary. The corner plot and predictive-check markdown cells
each state plainly, ahead of the code that produces them, what the reader
should conclude (wide or multimodal posterior at large t0 is the correct
result, not a failure; predictive agreement is a sanity check, not proof of
correctness), matching the required captions. lambda_median = 0.7208 is used
as a hardcoded constant with a comment, not read from results/lyapunov.txt,
since the notebook must not depend on anything outside
models/trained_model.keras, data/test.npz, and the three named code/ helpers.
Executed end to end with jupyter nbconvert --execute --inplace (the venv
already has its own registered "python3" kernelspec pointing at itself, so no
extra kernel setup was needed), then read the executed notebook's cell
outputs back with nbformat to check every cell for error outputs directly,
rather than trusting the nbconvert process exit code alone.

real commands executed:
  env_three_body\Scripts\python.exe (scratch energy-drift risk test)
  env_three_body\Scripts\python.exe (nbformat notebook builder script)
  env_three_body\Scripts\python.exe -m jupyter nbconvert --to notebook
    --execute --inplace --ExecutePreprocessor.timeout=300 results_notebook.ipynb
  env_three_body\Scripts\python.exe (read back all cell outputs, checked for errors)

result:
  all three named gates stayed clear: model loaded ("loaded
  models/trained_model.keras on cuda"), sample_posterior's signature matched
  exactly as expected, no re-simulation failures and no energy drift gate
  exceedance across all 30 re-simulated draws (15 per case, 2 cases) in this
  run; nbconvert exited 0 and a direct scan of every cell's outputs confirmed
  zero error-type outputs anywhere in the executed notebook
  device: cuda (RTX 5070 Ti Laptop GPU)
  small t0 case: test set index 1073, t0 = 0.000283 (raw), 0.000204 in units
  of 1/lambda (about as early as the observation window can start)
  large t0 case: test set index 516, t0 = 8.3222 (raw), 5.9987 in units of
  1/lambda (essentially the maximum, t0_max was 8.32447 = 6/lambda_median)
  total notebook runtime: 16.26 seconds
  results/posterior_example.png: the small-t0 posterior is tight and close to
  a single peak in every panel; the large-t0 posterior is visibly much wider
  in every parameter and shows non-Gaussian structure in several panels
  (for example a ring-like shape in the y1 versus x1 panel), a clear visual
  instance of the project's headline finding
  results/posterior_predictive.png: for both cases, most re-simulated
  trajectories cluster near the actual noisy observation points; the
  large-t0 panel's trajectories fan out much further away from the observed
  cluster (visually showing the same chaotic divergence the posterior width
  reflects), while still passing near the observed points, consistent with
  dynamically plausible posterior draws

problems (even solved ones): none in the final executed run; the marginal
energy drift exceedance seen during the earlier risk-scoping test did not
recur in the notebook's own random draws this time. This is inherently a
matter of chance (about 1.7 percent per draw measured on a small sample), so
a future re-run of this notebook could in principle still hit that gate; if
it does, the notebook will raise a clear RuntimeError naming the case and the
drift value, exactly as designed, rather than silently continuing.

decisions and why: kept the energy drift threshold and draw count as literal,
unmodified choices rather than tuning them to guarantee a clean run, since
changing either specifically to dodge the gate would have been improvising
around an explicit, named stop condition; ran the real notebook and let the
gate behave as designed instead. Verified success by reading the executed
notebook's own stored outputs back with nbformat rather than only trusting
the nbconvert exit code, since a partial failure inside one cell does not
necessarily make nbconvert itself exit non-zero in every configuration.

next step: waiting for the owner. Not started: docs/how_it_works.md.

## 2026-07-13, results notebook, two fixes to the posterior predictive cell

part of the project: results_notebook.ipynb only; train.py, diagnostics.py,
simulator.py, the data and the trained model were not touched, and nothing
was retrained

action taken: read the latest WORKLOG entry first. The owner's own arithmetic
on the previous entry's measured 1.7 percent per-draw exceedance rate showed
roughly a 40 percent chance the notebook's hard RuntimeError gate would crash
it on a re-run with 30 draws, unacceptable for a shareable deliverable. Made
exactly the two requested fixes to the posterior predictive cell and nothing
else.

Fix 1: removed the raise on energy drift exceedance. Every draw's max drift
is now recorded, exceedances above 1e-8 are counted, and both the max drift
over all draws and the exceedance count are printed per case. Plotting is
unconditional; only a genuine integrator failure (success False) still
raises. Added the requested markdown paragraph explaining why: the 1e-8 gate
was built for prior draws, which prior.py's rejection sampling filters for
escape and collision before they ever reach the dataset; posterior draws are
not filtered, so the posterior can honestly place mass near the accept/reject
boundary, and an occasional small drift exceedance there is information about
the posterior, not a bug.

Fix 2: each re-simulated trajectory is now split at t0 using a boolean mask
on the existing fine time grid (no extra integration calls): the segment from
0 to t0 is drawn dashed at alpha 0.08, the segment from t0 to t0 + T is drawn
solid at alpha 0.3. Only the first draw's segments carry a legend label
("pre-observation (not checked)" and "re-simulated (observation window)") so
the legend does not repeat per draw. Updated the same markdown cell to
explain which segment is the one actually being checked against the noisy
observation.

Also raised NUM_PREDICTIVE_DRAWS from 15 to 30 as requested, now that a
crash is no longer possible.

Hit and recovered from a real mistake while editing: the first attempt edited
the notebook in place with nbformat, read the old file fine, but the write
failed partway through with a UnicodeEncodeError (Windows' default cp1252
text encoding could not represent a character left over in the old cell
outputs, likely from a progress-bar glyph). Because Python's open(path, "w")
truncates the file immediately on open, the failed write left
results_notebook.ipynb at 0 bytes, destroying the previous version. Recovered
by reconstructing the notebook from scratch: the unaffected cells (0 through
8 and 11) were copied verbatim from this session's own earlier full read of
their source, and cells 9 and 10 were rebuilt with the two fixes; rewrote the
whole file in one pass with encoding="utf-8" explicit on the open() call this
time. Verified the rebuilt file is valid nbformat and contains the expected
content (the drift-explanation text, NUM_PREDICTIVE_DRAWS = 30, exactly one
remaining raise RuntimeError) before re-executing.

real commands executed:
  env_three_body\Scripts\python.exe (nbformat in-place edit attempt, failed
    with UnicodeEncodeError, truncated the file)
  env_three_body\Scripts\python.exe (nbformat full rebuild with utf-8, cells
    0-8 and 11 verbatim, 9 and 10 fixed)
  env_three_body\Scripts\python.exe -m jupyter nbconvert --to notebook
    --execute --inplace --ExecutePreprocessor.timeout=300 results_notebook.ipynb
  env_three_body\Scripts\python.exe (read back all cell outputs, checked for errors)

result:
  both named gates stayed clear: nbconvert exited 0, and a direct scan of
  every cell's stored outputs confirmed zero error-type outputs; both figures
  wrote with fresh timestamps
  small t0 case (index 1073): max energy drift over 30 draws 1.1197e-08, 1 of
  30 draws exceeded the 1e-8 reporting threshold; this is exactly the
  scenario fix 1 was written for, and it happened for real on this very run,
  the notebook printed it and kept going instead of crashing
  large t0 case (index 516): max energy drift over 30 draws 8.893e-09, 0 of
  30 exceeded
  total notebook runtime: 19.19 seconds
  results/posterior_predictive.png now clearly shows the solid,
  darker "observation window" segments clustering near the actual noisy
  observation points in both panels, with the faint dashed "pre-observation"
  segments visibly extending further away (most noticeably in the large t0
  panel, where one dashed line reaches out to roughly x = -13, y = -12
  before the plot ever gets to the part being checked), a much more honest
  picture than the previous single-style line that mixed observed and
  unobserved history together

problems (even solved ones): the UnicodeEncodeError file-truncation incident
described above, self-inflicted by not specifying an encoding on open() when
writing over the existing notebook file; recovered fully since this
session's own prior full read of the notebook's cell sources was still in
context, verified the rebuilt file before spending compute re-executing it.

decisions and why: reconstructed the full notebook in one rebuild script
rather than trying to patch the empty file, since a from-scratch rebuild
using known-good source text was safer and easier to verify correct than any
partial-recovery approach. Split the trajectory by boolean mask on the
already-computed fine grid rather than two separate integrate_system calls,
avoiding doubling the integration cost for a cosmetic change.

next step: waiting for the owner. Not started: docs/how_it_works.md.

## 2026-07-13, how_it_works.md and code style audit

part of the project: docs/how_it_works.md (new); a style audit of every file
in code/ and results_notebook.ipynb against CLAUDE.md's code style rules,
with cosmetic-only fixes applied. Wrote no code that produces results and
ran nothing that trains or regenerates data, as instructed.

action taken: read the latest WORKLOG entry and CLAUDE.md first, as always.

Task B (style audit) done first, since reading every file carefully for
compliance also builds the accurate picture needed for task A. Read
device.py, simulator.py, lyapunov.py, prior.py, make_dataset.py, train.py,
diagnostics.py in full, and every cell of results_notebook.ipynb, checking
each against CLAUDE.md's list: simple names, basic constructs only, no
classes unless a library forces it, no decorators, no clever one-liners, no
decorative separator lines, no emojis anywhere, short comments.

Found one real violation: results_notebook.ipynb, cell 10 (the posterior
predictive code cell), two lines used ternary conditional expressions
("pre_label = "..." if draw_index == 0 else None" and the same for
obs_label) to decide whether to attach a legend label on the first draw of
each case. This is exactly the kind of inline branching CLAUDE.md's minimal
list (def, for, if, print) does not include. Fixed by rewriting as an
explicit if/else block assigning the same two values, a byte-for-byte
behavior preserving change (Python guarantees identical semantics), so no
gate was triggered and no re-execution was needed to verify it. Deliberately
did not re-execute the notebook afterward, since the owner's instruction for
this turn was to write no code that produces results and run nothing that
regenerates data; the existing cell outputs and images were left untouched,
only the source text of that one cell changed.

Everything else in code/ was checked and found compliant, including two
things worth recording explicitly since they look like they might be
violations at a glance but are not: train.py's EpochLogger
(keras.callbacks.Callback subclass) is the one class in the whole codebase,
and it is justified in its own comment as the library-forces-it exception,
since keras callbacks are how keras itself hooks per-epoch behavior into
fit(); and train.py's chained bf.Adapter() builder calls are bayesflow's own
documented fluent API, not an invented one-liner. Simple list/generator
comprehensions used for parsing or joining comma separated text (lyapunov.py
and prior.py's results/lyapunov.txt read/write, for example) were reviewed
and judged not to be "clever one-liners" in the sense the rule targets,
since they are single-purpose, idiomatic, and already used consistently
throughout the codebase across many prior sessions without ever being
flagged; noted this judgment call explicitly rather than silently deciding
it. No decorative separator lines, no emojis, no absolute paths and no
advanced typing were found anywhere. Neither named gate fired (no file
needed a logic change to comply, and the one real fix was unambiguously
cosmetic), so nothing was escalated.

One unscoped but related observation, not a task B violation and not
touched: prior.py's COLLISION_RADIUS comment says "see WORKLOG.md for the
exploration," a comment that will dangle after WORKLOG.md is deleted at
project end. This is not a functional dependency (nothing reads that file)
and is outside the eight rules task B asked about, so it was left alone and
only noted here for the owner's awareness.

Task A (docs/how_it_works.md) written next, now that every code file and
the notebook had been read closely. Covers, in order: what the project
infers and why a wide posterior is the correct answer for a chaotic system
rather than a failure; every code/ file's inputs, outputs and the run order
from an empty repo to the final figures; every field of data/train.npz,
data/validation.npz, data/test.npz, data/dataset_info.txt,
models/trained_model.keras, models/training_info.txt and
models/training_log.txt / training_history.png; how the model works
conceptually (prior, simulator, noise, the t0 window design and why it
replaced the fixed-window design, the summary network, flow matching, what
amortized means); a one-subsection-each guide to reading
chaos_summary.png, recovery.png, contraction.png, calibration_ecdf.png,
posterior_example.png and posterior_predictive.png, each stating what a
good and a bad result look like; the two-design story (fixed window versus
t0 window, and the best-validation checkpoint fix, with the exact
calibration error numbers pulled from earlier WORKLOG entries: roughly 0.09
to 0.11 before the checkpoint fix, roughly 0.018 to 0.033 after); and the
two known limitations stated plainly (the aggregate SBC ECDF still sits
outside its 95 percent band even though per-bin calibration error is low,
and the roughly 1.7 percent posterior-draw energy drift exceedance rate).
Wrote this from figures and numbers already recorded in WORKLOG.md,
data_contracts.md and the actual files on disk, not from memory. Made two
cross-reference numbering mistakes while drafting (pointing at "section 7"
or "section 8" for content that actually landed in section 5 or 6 once the
document was fully written) and caught them with a grep over every "section
N" mention afterward, fixing all five wrong references.

Verified the hard constraint directly rather than assuming it: grepped
code/ for "how_it_works" (no matches) and scanned every notebook cell's
source for the same string (no matches). The file was never referenced
because nothing was written that reads it, so deleting it changes nothing,
exactly as required.

real commands executed:
  (Read on all seven code/ files)
  env_three_body\Scripts\python.exe (printed every notebook cell's source
    with per-cell line numbers, for the audit)
  env_three_body\Scripts\python.exe (applied the one cosmetic notebook fix:
    wrote to a temp file, validated it with nbformat, then os.replace'd the
    original only after the temp file checked out)
  env_three_body\Scripts\python.exe (verified the fix landed correctly and
    that cell outputs and execution_count were untouched)
  (Read on models/training_info.txt, data/dataset_info.txt,
    results/lyapunov.txt, for exact figures used in how_it_works.md)
  (Write on docs/how_it_works.md)
  (Grep for "how_it_works" across code/ and the notebook, confirming zero
    references)
  (Grep for "section N" across how_it_works.md, fixing 5 wrong
    cross-references)

result:
  task B violation list: exactly one violation (2 lines, 1 cell,
  results_notebook.ipynb cell 10), fixed. Nothing else found across all 7
  code/ files and all 12 notebook cells.
  task A: docs/how_it_works.md written, 7 numbered sections (what the
  project does; every code/ file and the run order; data files field by
  field; how the model works conceptually; how to read every figure; the
  two-design story; known limitations), all required content present,
  verified zero dependency from code/ or the notebook.

problems (even solved ones): the two cross-reference numbering mistakes in
the first draft of how_it_works.md, described above; caught and fixed by
grepping the finished document rather than trusting the draft.

decisions and why: judged simple, single-purpose list and generator
comprehensions as compliant "basic" style rather than "clever one-liners,"
since they are already the established, consistent idiom across this
codebase's whole history; flagged the judgment call explicitly rather than
silently applying it, so the owner can override it if they see it
differently. Did not re-execute the notebook after the one cosmetic fix,
since the fix is behavior-identical by construction and the owner's
instruction for this turn was to run nothing that produces results.

next step: waiting for the owner.

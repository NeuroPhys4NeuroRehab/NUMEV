function res = KalmanTrackingv4(file, opts)
%KALMANTRACKINGV4 Test whether EEG-NIRS coupling differs before, during and after tDCS.
%
%   res = KalmanTrackingv4(file) loads <file>.mat (mean_EEGpowResampFilt,
%   mean_nirsOxyFilt, tdcs_start; 10 Hz, already low-passed at 0.1 Hz),
%   decimates to 1 Hz (lossless for the 0.1 Hz band) and, for each requested
%   direction, models an output signal y from an input signal x with an ARX
%   model whose input-to-output impulse response is a sum of Gaussian basis
%   functions:
%
%       y(k) = c0 + sum_{i=1..p} a_i y(k-i) + sum_j c_j x_j(k) + e(k),
%       x_j(k) = sum_{l=L1..L2} G(l,j) x(k-l)
%
%   Directions (opts.direction):
%     'eeg2nirs'  x = log10 EEG band power, y = oxy-Hb   (EEG leads NIRS)
%     'nirs2eeg'  x = oxy-Hb, y = log10 EEG band power   (NIRS leads EEG)
%     'both'      both models, reported separately
%   Lags are strictly positive (L1 >= 1 s), so each direction captures only
%   the input leading the output; neither model has an instantaneous term.
%
%   Intercept and AR terms are always epoch-specific (nuisance); only the
%   input coefficients c are shared or separated between epochs, so tests
%   target coupling alone.
%
%   Epochs (time t relative to tdcs_start):
%     pre          t < 0
%     during       rampExcl <= t < tdcsDur                 (ramp-up excluded)
%       or, with duringSplit = s:
%     during early rampExcl <= t < s
%     during late  s <= t < tdcsDur
%     post         t >= tdcsDur + postGap                  (ramp-down excluded)
%   An epoch enters inference only if its effective number of independent
%   samples (Bartlett) is at least minNeff.
%
%   Inference per subject and direction
%     Presence  partial R^2 of the input terms within each epoch versus
%               surrogates whose input is phase-randomised within each epoch
%               (preserves its spectrum, destroys input-output alignment).
%     Change    T = (RSS_H0 - RSS_H1)/RSS_H1, H1 = epoch-specific coupling,
%               H0 = coupling shared by the epochs being compared. Null
%               distribution by fixed-design residual bootstrap from the H0
%               fit. Residuals are rescaled for degrees of freedom and
%               resampled within each epoch as Gaussian spectral surrogates
%               (observed Fourier amplitudes times complex normal weights),
%               which keeps the expected residual spectrum and epoch variance
%               and, unlike phase-only randomisation, the variability of the
%               low-frequency periodogram.
%               Simulated size at alpha = 0.05 (sub1 EEG/NIRS as input,
%               simulated output, early/late split): EEG->NIRS 2.5-5.5%
%               (200 null datasets); NIRS->EEG 4.5-5.2% (400 null datasets)
%               except 'during early vs pre' 8.0% and 'during late vs pre'
%               8.5%, whose calibrated thresholds are p < 0.030 and p < 0.039.
%               These figures depend on epoch lengths; re-check by simulation
%               for other recordings.
%     Effects   per-epoch partial R^2, impulse response h = G*c and its sum,
%               with percentile intervals from the same bootstrap under H1.
%   Descriptive
%     Time-varying impulse response from a linear Kalman filter with
%     random-walk parameters (process noise chosen by predictive likelihood),
%     and a sliding-window EEG-NIRS cross-correlation with pointwise and
%     family-wise surrogate thresholds (lag > 0: EEG lags NIRS).
%
%   res = KalmanTrackingv4(file, opts) overrides any default:
%     direction    'eeg2nirs' | 'nirs2eeg' | 'both'           'both'
%     fs           input sampling rate (Hz)                       10
%     decim        decimation factor                              10
%     tdcsDur      stimulation duration from tdcs_start (s)       600
%     rampExcl     excluded start of stimulation (s)              30
%     postGap      excluded start of post epoch (s)               30
%     duringSplit  split stimulation into early/late at t (s)     []
%     lagRange     input-to-output lags modelled (s)              [1 20]
%     basisCentres Gaussian basis centres (s)                     [3 8 13 18]
%     basisSD      Gaussian basis SD (s)                          2.5
%     arOrder      AR order ([] = BIC over 1..4)                  []
%     minNeff      minimum effective samples per epoch            20
%     nSur         surrogates for presence and xcorr tests        999
%     nBoot        bootstrap replicates for change tests / CIs    999
%     qGrid        process-noise / R grid for the descriptive KF  10.^(-6:0)
%                  (nuisance and input blocks chosen jointly; input also 0)
%     winXcorr     xcorr window (s)                               100
%     maxLagXcorr  xcorr max lag (s)                              20
%     seed         random seed                                    1
%     useParallel  run surrogates with parfor                     true
%     saveFigs     save figures as <file>_v4_*.jpg/.fig           false
%     makeFigs     draw figures                                   true
%
%   Results for a direction are in res.eeg2nirs / res.nirs2eeg; the
%   cross-correlation is in res.xcorr.
%
%   Limitations: there is no sham condition, so an epoch difference shows
%   that coupling changed, not that tDCS caused it (time, arousal, drift and
%   stimulation artefact in the EEG are not controlled). A lead in one
%   direction is temporal precedence, not causation: a common slow driver
%   with different delays produces the same pattern. Inference is per
%   subject; group results are a fixed-effect combination over these subjects.

if nargin < 2, opts = struct(); end
opts = applyDefaults(opts);
M = parallelWorkers(opts);

[u, z, t, fsd] = loadData(file, opts);
[ep, names] = epochLabels(t, opts);
res = struct('file', file, 'opts', opts, 'fs', fsd, 't', t, 'u', u, 'z', z, ...
    'epochLabel', ep, 'epochNames', {names});

dirs = directionList(opts.direction);
for d = 1:numel(dirs)
    if strcmp(dirs{d}, 'eeg2nirs')
        r = analyseDirection(u, z, ep, names, fsd, opts, M);
        r.input = 'EEG';   r.output = 'NIRS';
    else
        r = analyseDirection(z, u, ep, names, fsd, opts, M);
        r.input = 'NIRS';  r.output = 'EEG';
    end
    res.(dirs{d}) = r;
end
res.xcorr = xcorrSignificance(u, z, t, ep, fsd, opts, M);

if opts.makeFigs
    for d = 1:numel(dirs)
        saveFigures(plotDirection(res, res.(dirs{d})), file, ...
            {[dirs{d} '_epochs'], [dirs{d} '_tracking']}, opts);
    end
    saveFigures(plotXcorr(res), file, {'xcorr'}, opts);
end
end


function r = analyseDirection(x, y, ep, names, fsd, opts, M)
% Epoch-wise presence and change tests for input x -> output y.
K = numel(names);
[G, lagS] = gaussBasis(opts, fsd);
L2 = round(opts.lagRange(2) * fsd);
X  = basisRegressors(x, G, lagS, fsd);
neff = epochNeff(x, y, ep, K, L2);
est  = neff >= opts.minNeff;
E    = find(est)';

epochs = table(names(:), arrayfun(@(e) sum(ep == e), (1:K)'), neff, est, ...
    'VariableNames', {'epoch', 'nSamples', 'Neff', 'estimable'});
epochs.partialR2 = nan(K, 1);  epochs.partialR2ci = nan(K, 2);
epochs.pPresence = nan(K, 1);  epochs.hSum = nan(K, 1);  epochs.hSumCi = nan(K, 2);
r = struct('epochs', epochs, 'arOrder', NaN, 'lags', lagS, 'basis', G, ...
    'impulse', [], 'impulseCi', [], 'contrasts', [], 'tracking', []);
if isempty(E)
    warning('KalmanTrackingv4:noEpoch', 'No epoch has enough effective samples.');
    return;
end

p = opts.arOrder;
if isempty(p), p = selectArOrder(y, X, ep, E, L2); end
Yl   = lagMatrix(y, p);
rows = (1:numel(y))' > max(p, L2) & ismember(ep, E);
yy   = y(rows);
epr  = ep(rows);
Yr   = Yl(rows, :);
Xr   = X(rows, :);

[R2, C] = epochFits(Yr, Xr, yy, epr, E);
H = G * C;

seg   = blockIds(ep);
R2sur = zeros(numel(E), opts.nSur);
seed  = opts.seed;  nSur = opts.nSur;
parfor (s = 1:nSur, M)
    rs = RandStream('Threefry', 'Seed', seed);
    rs.Substream = s;
    xs = phaseRandBlocks(x, seg, 1, rs);
    Xs = basisRegressors(xs, G, lagS, fsd);
    R2sur(:, s) = epochFits(Yr, Xs(rows, :), yy, epr, E);
end
pPresence = (1 + sum(R2sur >= R2, 2)) / (nSur + 1);

rs = RandStream('Threefry', 'Seed', opts.seed);
rs.Substream = opts.nSur + 1;
D1  = designMatrix(Yr, Xr, epr, E, 1:numel(E));
[f1, e1] = olsFit(D1, yy);
Yb1 = f1 + gaussSpecBlocks(dfRescale(D1, e1, epr, E), epr, opts.nBoot, rs);
[R2b, Cb] = epochFits(Yr, Xr, Yb1, epr, E);
R2ci = pct(R2b, [2.5 97.5], 2);
Hb   = reshape(G * reshape(Cb, size(G, 2), []), numel(lagS), numel(E), []);
Hci  = pct(Hb, [2.5 97.5], 3);
hSum = sum(H, 1)';
hSumCi = pct(reshape(sum(Hb, 1), numel(E), []), [2.5 97.5], 2);

contrasts = changeContrasts(E, names);
for c = 1:numel(contrasts)
    D0 = designMatrix(Yr, Xr, epr, E, contrasts(c).grp);
    [f0, e0] = olsFit(D0, yy);
    T  = (rss(D0, yy) - rss(D1, yy)) / rss(D1, yy);
    rs.Substream = opts.nSur + 1 + c;
    Yb0 = f0 + gaussSpecBlocks(dfRescale(D0, e0, epr, E), epr, opts.nBoot, rs);
    Tb  = (rss(D0, Yb0) - rss(D1, Yb0)) ./ rss(D1, Yb0);
    contrasts(c).T = T;
    contrasts(c).p = (1 + sum(Tb >= T)) / (opts.nBoot + 1);
    if contrasts(c).isPair
        i1 = contrasts(c).pair(1);  i2 = contrasts(c).pair(2);
        contrasts(c).dR2   = R2(i2) - R2(i1);
        contrasts(c).dR2ci = pct(R2b(i2, :) - R2b(i1, :), [2.5 97.5]);
        dS = reshape(sum(Hb(:, i2, :) - Hb(:, i1, :), 1), 1, []);
        contrasts(c).dHsum   = hSum(i2) - hSum(i1);
        contrasts(c).dHsumCi = pct(dS, [2.5 97.5]);
    else
        contrasts(c).dR2 = NaN;  contrasts(c).dR2ci = [NaN NaN];
        contrasts(c).dHsum = NaN; contrasts(c).dHsumCi = [NaN NaN];
    end
end
if ~isempty(contrasts)
    pair = [contrasts.isPair];
    pHolm = nan(1, numel(contrasts));
    if any(pair), pHolm(pair) = holm([contrasts(pair).p]); end
    for c = 1:numel(contrasts), contrasts(c).pHolm = pHolm(c); end
end

kf = trackingKF(y, Yl, X, max(p, L2), opts.qGrid);
kf.h = G * kf.theta(:, end - size(G, 2) + 1:end)';
if kf.atGridEdge
    warning('KalmanTrackingv4:qGridEdge', 'KF process noise chosen at the edge of opts.qGrid; widen it.');
end

epochs.partialR2(E) = R2;        epochs.partialR2ci(E, :) = R2ci;
epochs.pPresence(E) = pPresence; epochs.hSum(E) = hSum;   epochs.hSumCi(E, :) = hSumCi;
r.epochs = epochs;     r.arOrder = p;        r.impulse = H;  r.impulseCi = Hci;
r.contrasts = contrasts;  r.tracking = kf;
end


function xc = xcorrSignificance(u, z, t, ep, fsd, opts, M)
% Sliding EEG-NIRS cross-correlation with EEG phase-randomised within epochs.
W = round(opts.winXcorr * fsd);
L = round(opts.maxLagXcorr * fsd);
[C, lags, tWin] = slidingXcorr(u, z, W, L, t);
stat = max(abs(C), [], 2);
seg  = blockIds(ep);
sur  = zeros(opts.nSur, numel(stat));
seed = opts.seed + 1;  nSur = opts.nSur;
parfor (s = 1:nSur, M)
    rs = RandStream('Threefry', 'Seed', seed);
    rs.Substream = s;
    us = phaseRandBlocks(u, seg, 1, rs);
    sur(s, :) = max(abs(slidingXcorr(us, z, W, L, t)), [], 2)';
end
xc = struct('C', C, 'lags', lags(:) / fsd, 'tWin', tWin, 'stat', stat, ...
    'p', (1 + sum(sur >= stat', 1))' / (nSur + 1), 'fwThreshold', pct(max(sur, [], 2), 95));
end


function dirs = directionList(direction)
switch direction
    case 'both', dirs = {'eeg2nirs', 'nirs2eeg'};
    case {'eeg2nirs', 'nirs2eeg'}, dirs = {direction};
    otherwise, error('KalmanTrackingv4:direction', 'direction must be eeg2nirs, nirs2eeg or both.');
end
end


function [u, z, t, fsd] = loadData(file, opts)
S  = load([file '.mat'], 'mean_EEGpowResampFilt', 'mean_nirsOxyFilt', 'tdcs_start');
t0 = double(S.tdcs_start);
D  = opts.decim;
first = mod(t0 - 1, D) + 1;
u  = double(S.mean_EEGpowResampFilt(first:D:end));
z  = double(S.mean_nirsOxyFilt(first:D:end));
u  = (u(:) - mean(u)) / std(u);
z  = (z(:) - mean(z)) / std(z);
fsd = opts.fs / D;
k0 = (t0 - first) / D + 1;
t  = ((1:numel(z))' - k0) / fsd;
end


function [ep, names] = epochLabels(t, opts)
if isempty(opts.duringSplit)
    names = {'pre', 'during', 'post'};
    edges = [-inf 0; opts.rampExcl opts.tdcsDur; opts.tdcsDur + opts.postGap inf];
else
    names = {'pre', 'during early', 'during late', 'post'};
    edges = [-inf 0; opts.rampExcl opts.duringSplit; opts.duringSplit opts.tdcsDur; ...
        opts.tdcsDur + opts.postGap inf];
end
ep = zeros(size(t));
for e = 1:numel(names)
    ep(t >= edges(e, 1) & t < edges(e, 2)) = e;
end
end


function seg = blockIds(ep)
% Epochs keep their label; each excluded gap becomes its own negative block.
seg = ep;
gap = ep == 0;
run = cumsum(gap & [true; ~gap(1:end - 1)]);
seg(gap) = -run(gap);
end


function [G, lagS] = gaussBasis(opts, fsd)
lagS = (round(opts.lagRange(1) * fsd):round(opts.lagRange(2) * fsd))' / fsd;
G = exp(-0.5 * ((lagS - opts.basisCentres(:)') / opts.basisSD).^2);
G = G ./ sum(G, 1);
end


function X = basisRegressors(u, G, lagS, fsd)
k = round(lagS * fsd);
X = zeros(numel(u), size(G, 2));
for j = 1:size(G, 2)
    kernel = zeros(max(k) + 1, 1);
    kernel(k + 1) = G(:, j);
    X(:, j) = filter(kernel, 1, u);
end
end


function Zl = lagMatrix(z, p)
Zl = zeros(numel(z), p);
for i = 1:p
    Zl(i + 1:end, i) = z(1:end - i);
end
end


function neff = epochNeff(u, z, ep, K, L2)
% Bartlett effective sample size for the correlation of two autocorrelated series.
neff = zeros(K, 1);
for e = 1:K
    idx = find(ep == e);
    idx = idx(idx > L2);
    n = numel(idx);
    if n < 8, continue; end
    ru = acf(u(idx));  rz = acf(z(idx));
    k = floor(n / 4);
    neff(e) = min(n, max(1, n / (1 + 2 * sum(ru(2:k + 1) .* rz(2:k + 1)))));
end
end


function r = acf(x)
x = x - mean(x);
n = numel(x);
f = fft(x, 2^nextpow2(2 * n));
r = ifft(abs(f).^2, 'symmetric');
r = r(1:n) / r(1);
end


function p = selectArOrder(z, X, ep, E, L2)
bic = inf(4, 1);
for q = 1:4
    Zl = lagMatrix(z, q);
    rows = (1:numel(z))' > max(4, L2) & ismember(ep, E);
    D = designMatrix(Zl(rows, :), X(rows, :), ep(rows), E, 1:numel(E));
    n = sum(rows);
    bic(q) = n * log(rss(D, z(rows)) / n) + size(D, 2) * log(n);
end
[~, p] = min(bic);
end


function D = designMatrix(Zl, X, ep, E, grp)
% Epoch-specific intercept and AR terms; EEG terms shared within each group.
blocks = cell(1, numel(E) + max(grp));
for i = 1:numel(E)
    m = double(ep == E(i));
    blocks{i} = [m, m .* Zl];
end
for g = 1:max(grp)
    m = double(ismember(ep, E(grp == g)));
    blocks{numel(E) + g} = m .* X;
end
D = [blocks{:}];
end


function [R2, C] = epochFits(Zl, X, Y, ep, E)
% Per-epoch partial R^2 of the EEG terms and their coefficients, for every column of Y.
nb = size(X, 2);
R2 = zeros(numel(E), size(Y, 2));
C  = zeros(nb, numel(E), size(Y, 2));
for i = 1:numel(E)
    r  = ep == E(i);
    Dr = [ones(sum(r), 1), Zl(r, :)];
    Df = [Dr, X(r, :)];
    B  = Df \ Y(r, :);
    R2(i, :) = 1 - sum((Y(r, :) - Df * B).^2, 1) ./ rss(Dr, Y(r, :));
    C(:, i, :) = reshape(B(end - nb + 1:end, :), nb, 1, []);
end
end


function [f, e] = olsFit(D, y)
f = D * (D \ y);
e = y - f;
end


function s = rss(D, Y)
[Q, ~] = qr(D, 0);
s = sum((Y - Q * (Q' * Y)).^2, 1);
end


function contrasts = changeContrasts(E, names)
% Omnibus (all epochs share coupling) when 3 epochs are estimable; with 2 it
% equals the single pairwise contrast. Pairwise: one pair shares coupling.
contrasts = struct('name', {}, 'pair', {}, 'grp', {}, 'isPair', {});
if numel(E) < 2, return; end
if numel(E) >= 3
    contrasts(end + 1) = struct('name', 'omnibus', 'pair', 1:numel(E), ...
        'grp', ones(1, numel(E)), 'isPair', false);
end
for a = 1:numel(E) - 1
    for b = a + 1:numel(E)
        grp = 1:numel(E);
        grp(b) = a;
        contrasts(end + 1) = struct('name', [names{E(b)} ' vs ' names{E(a)}], 'pair', [a b], ...
            'grp', renumber(grp), 'isPair', true); %#ok<AGROW>
    end
end
end


function out = renumber(g)
% Relabel groups 1..K in order of first appearance.
seen = [];
out  = zeros(size(g));
for i = 1:numel(g)
    k = find(seen == g(i), 1);
    if isempty(k)
        seen(end + 1) = g(i); %#ok<AGROW>
        k = numel(seen);
    end
    out(i) = k;
end
end


function Y = phaseRandBlocks(x, seg, nRep, rs)
% Phase-randomised surrogates of x, independently within each block of seg.
Y = zeros(numel(x), nRep);
for g = unique(seg(:))'
    idx = find(seg == g);
    n = numel(idx);
    if n < 3, Y(idx, :) = repmat(x(idx), 1, nRep); continue; end
    F  = fft(x(idx));
    ph = zeros(n, nRep);
    pos = 2:ceil(n / 2);
    ph(pos, :) = 2 * pi * rand(rs, numel(pos), nRep);
    ph(n - pos + 2, :) = -ph(pos, :);
    Y(idx, :) = ifft(abs(F) .* exp(1i * (angle(F) + ph)), [], 1, 'symmetric');
end
end


function e = dfRescale(D, e, ep, E)
% Inflate residuals by sqrt(n/(n-k)) within each epoch, k = parameters fitted there.
for i = 1:numel(E)
    m = ep == E(i);
    k = rank(D(m, any(D(m, :) ~= 0, 1)));
    e(m) = e(m) * sqrt(sum(m) / (sum(m) - k));
end
end


function Y = gaussSpecBlocks(x, seg, nRep, rs)
% Gaussian spectral surrogates of x within each block of seg: observed Fourier
% amplitudes times complex normal weights (Hermitian), block mean preserved.
Y = zeros(numel(x), nRep);
for g = unique(seg(:))'
    idx = find(seg == g);
    n = numel(idx);
    if n < 3, Y(idx, :) = repmat(x(idx), 1, nRep); continue; end
    A = abs(fft(x(idx)));
    W = complex(randn(rs, n, nRep), randn(rs, n, nRep)) / sqrt(2);
    pos = 2:ceil(n / 2);
    W(n - pos + 2, :) = conj(W(pos, :));
    W(1, :) = 1;
    if mod(n, 2) == 0, W(n / 2 + 1, :) = randn(rs, 1, nRep); end
    Yb = ifft(A .* W, [], 1, 'symmetric');
    Y(idx, :) = Yb - mean(Yb, 1) + mean(x(idx));
end
end


function q = pct(x, p, dim)
% Percentiles (linear interpolation, as MATLAB prctile) without the Statistics Toolbox.
if nargin < 3, dim = find(size(x) > 1, 1); if isempty(dim), dim = 1; end, end
perm = [dim, setdiff(1:ndims(x), dim)];
x  = permute(x, perm);
sz = size(x);
n  = sz(1);
x  = sort(reshape(x, n, []), 1);
pos = min(max(p(:) / 100 * n + 0.5, 1), n);
lo = floor(pos);  hi = ceil(pos);  w = pos - lo;
q  = x(lo, :) .* (1 - w) + x(hi, :) .* w;
q  = ipermute(reshape(q, [numel(p), sz(2:end)]), perm);
end


function p = holm(p)
[ps, i] = sort(p);
m = numel(p);
adj = min(1, cummax((m - (0:m - 1)) .* ps));
p(i) = adj;
end


function kf = trackingKF(z, Zl, X, k0, qGrid)
% Linear KF on [intercept, AR | EEG basis] coefficients as random walks, with
% separate process noise for the nuisance and EEG blocks chosen jointly by
% the innovation log-likelihood. dLL compares time-varying with constant
% EEG coupling (qEEG = 0), each with its best nuisance noise.
Phi = [ones(numel(z), 1), Zl, X];
rows = (k0 + 1:numel(z))';
R  = rss(Phi(rows, :), z(rows)) / (numel(rows) - size(Phi, 2));
nn = 1 + size(Zl, 2);
qE = [0, qGrid];
LL = -inf(numel(qGrid), numel(qE));
for i = 1:numel(qGrid)
    for j = 1:numel(qE)
        qv = [repmat(qGrid(i), nn, 1); repmat(qE(j), size(X, 2), 1)] * R;
        [~, ~, LL(i, j)] = kfRun(Phi, z, rows, R, qv);
    end
end
[~, best] = max(LL(:));
[i, j] = ind2sub(size(LL), best);
qv = [repmat(qGrid(i), nn, 1); repmat(qE(j), size(X, 2), 1)] * R;
[TH, zPred] = kfRun(Phi, z, rows, R, qv);
kf = struct('theta', TH, 'zPred', zPred, 'qNuisance', qGrid(i), 'qEEG', qE(j), ...
    'dLL', LL(i, j) - max(LL(:, 1)), 'atGridEdge', i == numel(qGrid) || j == numel(qE), 'LL', LL);
end


function [TH, zp, ll] = kfRun(Phi, z, rows, R, q)
n = size(Phi, 2);
th = zeros(n, 1);  P = 10 * eye(n);  Q = diag(q);
TH = nan(numel(z), n);  zp = nan(numel(z), 1);  ll = 0;
for k = rows'
    P = P + Q;
    ph = Phi(k, :)';
    zp(k) = ph' * th;
    Pph = P * ph;
    S = ph' * Pph + R;
    K = Pph / S;
    v = z(k) - zp(k);
    th = th + K * v;
    P = P - K * Pph';
    ll = ll - 0.5 * (log(2 * pi * S) + v^2 / S);
    TH(k, :) = th';
end
end


function [C, lags, tWin] = slidingXcorr(u, z, W, maxLag, t)
% Demeaned, normalised cross-correlation of u and z in every window of W samples.
nWin = numel(u) - W + 1;
nfft = 2^nextpow2(W + maxLag);
idx  = (1:nWin)' + (0:W - 1);
Wu = u(idx);  Wz = z(idx);
Wu = Wu - mean(Wu, 2);  Wz = Wz - mean(Wz, 2);
R = ifft(fft(Wu, nfft, 2) .* conj(fft(Wz, nfft, 2)), [], 2, 'symmetric');
C = [R(:, nfft - maxLag + 1:nfft), R(:, 1:maxLag + 1)] ./ sqrt(sum(Wu.^2, 2) .* sum(Wz.^2, 2));
lags = -maxLag:maxLag;
tWin = t(1:nWin) + (W - 1) / 2 * (t(2) - t(1));
end


function figs = plotDirection(res, r)
names = res.epochNames;
E   = find(r.epochs.estimable)';
col = lines(numel(names));
t   = res.t;

figs(1) = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
tiledlayout(2, 2, 'TileSpacing', 'compact');

nexttile([1 2]);
plot(t, res.z, 'b', t, res.u, 'k');
hold on;
yl = ylim;
for e = 1:numel(names)
    m = res.epochLabel == e;
    if any(m)
        patch([min(t(m)) max(t(m)) max(t(m)) min(t(m))], yl([1 1 2 2]), col(e, :), ...
            'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
hold off;
xlabel('Time from tDCS onset (s)');
ylabel('z-score');
legend('NIRS oxy-Hb', 'EEG band power');
title(sprintf('%s \\rightarrow %s', r.input, r.output));
box on;

nexttile;
hold on;
for i = 1:numel(E)
    e = E(i);
    fill([r.lags; flipud(r.lags)], [r.impulseCi(:, i, 1); flipud(r.impulseCi(:, i, 2))], ...
        col(e, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(r.lags, r.impulse(:, i), 'Color', col(e, :), 'LineWidth', 2, 'DisplayName', names{e});
end
yline(0, 'k:', 'HandleVisibility', 'off');
hold off;
xlabel(sprintf('Lag (s), %s leads %s', r.input, r.output));
ylabel('Impulse response (z/z)');
legend('Location', 'best');
box on;

nexttile;
ep = r.epochs;
hold on;
for i = 1:numel(E)
    e = E(i);
    bar(i, ep.partialR2(e), 'FaceColor', col(e, :));
    errorbar(i, ep.partialR2(e), ep.partialR2(e) - ep.partialR2ci(e, 1), ...
        ep.partialR2ci(e, 2) - ep.partialR2(e), 'k');
    text(i, ep.partialR2ci(e, 2), sprintf('p=%.3f', ep.pPresence(e)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
hold off;
set(gca, 'XTick', 1:numel(E), 'XTickLabel', names(E));
ylabel(sprintf('Partial R^2 of %s terms', r.input));
box on;

figs(2) = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
imagesc(t, r.lags, r.tracking.h);
axis xy;
h  = abs(r.tracking.h(isfinite(r.tracking.h)));
cl = pct(h(:), 99);
if isempty(cl) || ~isfinite(cl) || cl <= 0, cl = max([h(:); eps]); end
set(gca, 'CLim', [-cl cl]);
colormap(gca, blueWhiteRed(256));
colorbar;
epochLines(res);
xlabel('Time from tDCS onset (s)');
ylabel(sprintf('Lag (s), %s leads %s', r.input, r.output));
title(sprintf('Time-varying %s\\rightarrow%s impulse response (KF; q/R nuisance %g, input %g; \\DeltaLL vs constant = %.1f)', ...
    r.input, r.output, r.tracking.qNuisance, r.tracking.qEEG, r.tracking.dLL));
end


function fig = plotXcorr(res)
fig = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
xc = res.xcorr;
imagesc(xc.tWin, xc.lags, xc.C');
axis xy;
set(gca, 'CLim', [-1 1]);
colormap(gca, blueWhiteRed(256));
colorbar;
hold on;
sig = xc.p < 0.05;
plot(xc.tWin(sig), repmat(max(xc.lags), sum(sig), 1), 'k.', 'MarkerSize', 6);
fw = xc.stat > xc.fwThreshold;
plot(xc.tWin(fw), repmat(max(xc.lags), sum(fw), 1), 'g*', 'MarkerSize', 6);
hold off;
epochLines(res);
xlabel('Window centre, time from tDCS onset (s)');
ylabel('Lag (s), > 0: EEG lags NIRS');
title('Sliding EEG-NIRS cross-correlation (dots: p < 0.05 per window; stars: family-wise p < 0.05)');
end


function epochLines(res)
o = res.opts;
x = [0, o.rampExcl, o.duringSplit, o.tdcsDur, o.tdcsDur + o.postGap];
xline(x, 'k--', 'HandleVisibility', 'off');
end


function saveFigures(figs, file, tags, opts)
if ~opts.saveFigs, return; end
for n = 1:numel(figs)
    print(figs(n), sprintf('%s_v4_%s.jpg', file, tags{n}), '-djpeg', '-r300');
    savefig(figs(n), sprintf('%s_v4_%s.fig', file, tags{n}));
end
end


function cmap = blueWhiteRed(n)
h    = linspace(-1, 1, n)';
cmap = [min(1, 1 + h), 1 - abs(h), min(1, 1 - h)];
end


function M = parallelWorkers(opts)
M = 0;
if ~opts.useParallel || ~license('test', 'Distrib_Computing_Toolbox'), return; end
try
    pool = gcp('nocreate');
    if isempty(pool)
        cluster = parcluster;
        pool = parpool(cluster, cluster.NumWorkers);
    end
    M = pool.NumWorkers;
catch
    M = 0;
end
end


function opts = applyDefaults(opts)
def = struct('direction', 'both', 'fs', 10, 'decim', 10, 'tdcsDur', 600, 'rampExcl', 30, ...
    'postGap', 30, 'duringSplit', [], ...
    'lagRange', [1 20], 'basisCentres', [3 8 13 18], 'basisSD', 2.5, 'arOrder', [], ...
    'minNeff', 20, 'nSur', 999, 'nBoot', 999, 'qGrid', 10.^(-6:0), ...
    'winXcorr', 100, 'maxLagXcorr', 20, 'seed', 1, 'useParallel', true, ...
    'saveFigs', false, 'makeFigs', true);
names = fieldnames(def);
for n = 1:numel(names)
    if ~isfield(opts, names{n})
        opts.(names{n}) = def.(names{n});
    end
end
end

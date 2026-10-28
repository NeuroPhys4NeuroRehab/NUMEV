function [summary, group, results] = runTDCSCoupling(files, opts)
%RUNTDCSCOUPLING Run KalmanTrackingv4 on every subject and combine the results.
%
%   [summary, group, results] = runTDCSCoupling({'sub1','sub2',...}, opts)
%   returns one row per subject, direction and test (summary), a group table
%   (group), and the per-subject result structs. Both tables are also written
%   to tdcs_coupling_summary.csv and tdcs_coupling_group.csv.
%
%   Example, both directions with stimulation split into early and late:
%     opts = struct('direction', 'both', 'duringSplit', 315);
%     [summary, group] = runTDCSCoupling({'sub1','sub2','sub3','sub4','sub5'}, opts);
%
%   Group inference is a fixed-effect combination over these subjects:
%     Stouffer's Z on per-subject p-values (one-sided, small p = evidence),
%     and the sign count of the epoch difference in partial R^2. With five
%     subjects an exact two-sided sign test cannot fall below p = 0.0625, so
%     the sign count is descriptive only.

if nargin < 2, opts = struct(); end
results = cell(numel(files), 1);
for i = 1:numel(files)
    fprintf('%s ...\n', files{i});
    results{i} = KalmanTrackingv4(files{i}, opts);
end

dirs = intersect({'eeg2nirs', 'nirs2eeg'}, fieldnames(results{1}), 'stable');
rows = {};
for i = 1:numel(files)
    for d = 1:numel(dirs)
        r = results{i}.(dirs{d});
        for e = 1:height(r.epochs)
            ep = r.epochs(e, :);
            rows(end + 1, :) = {files{i}, dirs{d}, ['presence: ' ep.epoch{1}], ep.Neff, ep.estimable, ...
                ep.partialR2, ep.partialR2ci(1), ep.partialR2ci(2), NaN, ep.pPresence, NaN}; %#ok<AGROW>
        end
        for c = 1:numel(r.contrasts)
            k = r.contrasts(c);
            rows(end + 1, :) = {files{i}, dirs{d}, ['change: ' k.name], NaN, true, ...
                k.dR2, k.dR2ci(1), k.dR2ci(2), k.T, k.p, k.pHolm}; %#ok<AGROW>
        end
    end
end
summary = cell2table(rows, 'VariableNames', {'subject', 'direction', 'test', 'Neff', 'estimable', ...
    'effect', 'ciLow', 'ciHigh', 'T', 'p', 'pHolm'});

valid = summary.estimable & ~isnan(summary.p);
keys  = unique(strcat(summary.direction(valid), '|', summary.test(valid)), 'stable');
g = cell(numel(keys), 7);
for j = 1:numel(keys)
    parts = strsplit(keys{j}, '|');
    m  = valid & strcmp(summary.direction, parts{1}) & strcmp(summary.test, parts{2});
    pv = summary.p(m);
    Z  = sum(sqrt(2) * erfcinv(2 * pv)) / sqrt(numel(pv));
    g(j, :) = {parts{1}, parts{2}, sum(m), Z, 0.5 * erfc(Z / sqrt(2)), ...
        sum(summary.effect(m) > 0), strjoin(summary.subject(m)', ',')};
end
group = cell2table(g, 'VariableNames', {'direction', 'test', 'nSubjects', 'stoufferZ', ...
    'pGroup', 'nPositiveEffect', 'subjects'});

writetable(summary, 'tdcs_coupling_summary.csv');
writetable(group, 'tdcs_coupling_group.csv');
disp(summary);
disp(group);
end

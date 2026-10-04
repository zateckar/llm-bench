/* One shared workload and coordinate keeps every run column aligned. */
function performanceComparison(data) {
    return {
        data, workloadId: '', context: null, concurrency: null, metric: '', byRow: {},
        init() {
            this.workloadId = data.workloads[0].id;
            this.changeWorkload();
        },
        get workload() { return data.workloads.find(w => w.id === this.workloadId); },
        get metricInfo() { return this.workload.metrics.find(m => m.key === this.metric); },
        changeWorkload() {
            const previousContext = this.context, previousConcurrency = this.concurrency;
            this.byRow = {};
            for (const row of this.workload.rows) this.byRow[`${row.context}/${row.concurrency}`] = row.points;
            this.context = this.workload.contexts.includes(previousContext) ? previousContext : this.workload.contexts[0];
            this.concurrency = this.workload.concurrencies.includes(previousConcurrency) ? previousConcurrency : this.workload.concurrencies[0];
            if (!this.workload.metrics.some(m => m.key === this.metric)) this.metric = this.workload.metrics[0].key;
        },
        points(context = this.context, concurrency = this.concurrency) {
            return this.byRow[`${context}/${concurrency}`] || this.data.runs.map(() => null);
        },
        value(point, key = this.metric) { return point ? point.formatted[key] : '—'; },
        status(point) {
            if (!point) return 'Not measured for this workload';
            return ({measured: 'Measured', failed: 'Failed', incomplete: 'Incomplete answer',
                unsupported_context: 'Context rejected', skipped_context: 'Context skipped',
                skipped_failure: 'Load skipped', priming_failed: 'Cache priming failed'})[point.status] || point.status;
        },
        problem(point) { return point && (point.errors > 0 || point.incomplete > 0 || point.status !== 'measured'); },
        title(point) { return [this.status(point), point?.samples, point?.failures, point?.error].filter(Boolean).join(' · '); }
    };
}

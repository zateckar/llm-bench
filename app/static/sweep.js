/* Desktop explorer for one bounded context/concurrency matrix at a time. */
function sweepExplorer(data) {
    return {
        data, byCell: {}, effort: 'default', metric: 'first_p50', context: 256, concurrency: 1,
        minimum: null, maximum: null,
        init() {
            for (const cell of data.cells) this.byCell[`${cell.effort}/${cell.context}/${cell.concurrency}`] = cell;
            this.effort = data.efforts.find(e => e.status === 'accepted')?.effort || data.efforts[0]?.effort || 'default';
            this.context = data.contexts[0]; this.concurrency = data.concurrencies[0];
            this.refresh();
        },
        get metricInfo() { return data.metrics.find(m => m.key === this.metric); },
        get profile() { return data.efforts.find(e => e.effort === this.effort); },
        get selected() { return this.cell(this.context, this.concurrency); },
        cell(context, concurrency) {
            return this.byCell[`${this.effort}/${context}/${concurrency}`] || {
                context, concurrency, effort: this.effort,
                status: this.profile?.status === 'accepted' ? 'not_measured' : this.profile?.status || 'not_probed',
                values: {}, requests: 0, completed: 0, errors: 0, incomplete: 0,
                error: this.profile?.error || '', estimated: 0, estimated_input: 0, bursts: 0, cached: 0, ttft_count: 0
            };
        },
        refresh() {
            const values = data.cells.filter(c => c.effort === this.effort).map(c => c.values[this.metric]).filter(v => typeof v === 'number' && Number.isFinite(v));
            this.minimum = values.length ? Math.min(...values) : null;
            this.maximum = values.length ? Math.max(...values) : null;
        },
        select(context, concurrency) { this.context = context; this.concurrency = concurrency; this.revealSelection(); },
        revealSelection() {
            this.$nextTick(() => {
                const button = this.$el.querySelector('.sweep-selected'), scroller = this.$el.querySelector('.sweep-map-scroll');
                if (!button || !scroller) return;
                const tile = button.getBoundingClientRect(), box = scroller.getBoundingClientRect();
                if (tile.top < box.top + 36 || tile.bottom > box.bottom) scroller.scrollTop += tile.top - box.top - 40;
                if (tile.left < box.left + 95 || tile.right > box.right) scroller.scrollLeft += tile.left - box.left - 100;
            });
        },
        statusLabel(status) { return ({measured:'Measured', unsupported_context:'Context rejected', skipped_context:'Context skipped', skipped_failure:'Load skipped', failed:'Failed', incomplete:'Incomplete answer', not_measured:'Not measured', not_probed:'Not probed', accepted:'Accepted', unsupported:'Unsupported effort', probe_failed:'Probe failed'})[status] || status; },
        format(value, unit = this.metricInfo.unit) {
            if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
            if (unit !== 'ms') return value.toLocaleString(undefined, {maximumFractionDigits:1}) + ' ' + unit;
            if (value >= 60000) return (value / 60000).toFixed(2) + ' min';
            if (value >= 1000) return (value / 1000).toFixed(2) + ' s';
            return Math.round(value).toLocaleString() + ' ms';
        },
        color(cell) {
            const value = cell.values[this.metric];
            if (typeof value === 'number' && Number.isFinite(value)) {
                const ratio = this.maximum > this.minimum ? (value - this.minimum) / (this.maximum - this.minimum) : 0.5;
                return `hsl(${215 + ratio * 55},70%,${75 - ratio * 35}%)`;
            }
            if (['failed','unsupported_context','incomplete'].includes(cell.status)) return '#a55a37';
            return 'var(--sweep-empty, #33333d)';
        },
        foreground(cell) {
            const value = cell.values[this.metric];
            const ratio = this.maximum > this.minimum ? (value - this.minimum) / (this.maximum - this.minimum) : 0.5;
            return typeof value === 'number' && Number.isFinite(value) && ratio < 0.45 ? '#111827' : '#fff';
        },
        cellLabel(cell) {
            const value = cell.values[this.metric];
            if (typeof value === 'number' && Number.isFinite(value)) return this.format(value);
            if (['failed', 'unsupported_context'].includes(cell.status)) return 'Failed';
            if (cell.status === 'incomplete') return 'Incomplete';
            if (cell.status.startsWith('skipped_')) return 'Skipped';
            return cell.requests ? 'n/a' : '—';
        },
        title(cell) {
            return `${cell.context.toLocaleString()} tokens · c=${cell.concurrency} · ${this.format(cell.values[this.metric])} · ${this.statusLabel(cell.status)} · ${cell.completed}/${cell.requests} complete`;
        },
        slice(axis) {
            return (axis === 'context' ? data.contexts : data.concurrencies).map(n => this.cell(axis === 'context' ? n : this.context, axis === 'context' ? this.concurrency : n));
        },
        chart(axis) {
            const cells = this.slice(axis);
            const points = cells.map(c => [axis === 'context' ? c.context : c.concurrency, c.values[this.metric]]);
            const valid = points.filter(p => typeof p[1] === 'number' && Number.isFinite(p[1]));
            if (!valid.length) return '<p class="perf-caption sweep-no-data">No measurements for this slice.</p>';
            const esc = s => String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
            const maxX = Math.max(...points.map(p => p[0])), minX = Math.min(...points.map(p => p[0]));
            const maxY = Math.max(...valid.map(p => p[1])) || 1;
            const x = v => 72 + (v - minX) / (maxX - minX || 1) * 464;
            const y = v => 196 - v / maxY * 170;
            let svg = '<svg viewBox="0 0 560 240" role="img" aria-label="' + esc(this.metricInfo.label + ' by ' + axis) + '"><g font-size="10" fill="currentColor">';
            for (let i=0; i<=4; i++) {
                const value = maxY * i / 4, yy = y(value);
                svg += `<line x1="72" x2="536" y1="${yy}" y2="${yy}" stroke="currentColor" opacity=".12"/><text x="65" y="${yy+3}" text-anchor="end">${esc(this.format(value))}</text>`;
            }
            for (const value of [...new Set([minX, Math.round((minX+maxX)/2), maxX])]) svg += `<text x="${x(value)}" y="215" text-anchor="middle">${esc(value.toLocaleString())}</text>`;
            svg += `<text x="304" y="234" text-anchor="middle">${axis === 'context' ? 'Input reference tokens' : 'Concurrent requests'}</text></g>`;
            let segment = [];
            const flush = () => { if (segment.length > 1) svg += `<polyline points="${segment.join(' ')}" fill="none" stroke="#609bff" stroke-width="2"/>`; segment=[]; };
            for (const [xx, value] of points) {
                if (typeof value !== 'number' || !Number.isFinite(value)) { flush(); continue; }
                segment.push(`${x(xx)},${y(value)}`);
                svg += `<circle cx="${x(xx)}" cy="${y(value)}" r="3" fill="#609bff"><title>${esc(xx.toLocaleString() + ': ' + this.format(value))}</title></circle>`;
            }
            flush(); return svg + '</svg>';
        }
    };
}

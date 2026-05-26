import { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";
import { api } from "../api/client";

export default function OHLCChart({ ticker = "AAPL" }) {
  const containerRef = useRef(null);
  const chartRef     = useRef(null);
  const seriesRef    = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: { background: { color: "#161b22" }, textColor: "#8b949e" },
      grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: "#30363d" },
      timeScale: { borderColor: "#30363d", timeVisible: false },
      width: containerRef.current.clientWidth,
      height: 280,
    });
    chartRef.current = chart;
    const series = chart.addCandlestickSeries({
      upColor: "#3fb950", downColor: "#f85149",
      borderUpColor: "#3fb950", borderDownColor: "#f85149",
      wickUpColor: "#3fb950", wickDownColor: "#f85149",
    });
    seriesRef.current = series;

    api.getPrices(ticker, 90)
      .then((rows) => {
        if (rows && rows.length) {
          series.setData(rows);
          chart.timeScale().fitContent();
        }
      })
      .catch(() => {});

    const ro = new ResizeObserver(() => {
      if (containerRef.current)
        chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    ro.observe(containerRef.current);

    return () => { ro.disconnect(); chart.remove(); };
  }, [ticker]);

  return <div ref={containerRef} className="w-full" />;
}

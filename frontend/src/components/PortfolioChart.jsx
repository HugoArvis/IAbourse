import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { api } from "../api/client";

export default function PortfolioChart() {
  const [data, setData] = useState([]);

  useEffect(() => {
    api.getPortfolioHistory()
      .then((rows) => {
        if (rows && rows.length) {
          setData(rows.map((r) => ({ day: r.time, value: r.value })));
        }
      })
      .catch(() => {});
  }, []);

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis dataKey="day" tick={{ fill: "#8b949e", fontSize: 11 }} tickFormatter={(v) => v.slice(5)} />
        <YAxis tick={{ fill: "#8b949e", fontSize: 11 }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} domain={["auto", "auto"]} />
        <Tooltip
          contentStyle={{ background: "#161b22", border: "1px solid #30363d", color: "#e6edf3" }}
          formatter={(v) => [`$${Number(v).toLocaleString()}`, "Portfolio"]}
        />
        <Line type="monotone" dataKey="value" stroke="#58a6ff" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, RadarChart, Radar,
  PolarGrid, PolarAngleAxis,
} from "recharts";
import Card from "../components/Card";
import styles from "./Metrics.module.css";

const METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"];
const COLORS = { "XGBoost": "#003580", "LightGBM": "#c8102e", "CatBoost": "#f59e0b" };

export default function Metrics() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/metrics/")
      .then(async (res) => {
        const body = await res.text();
        if (!res.ok) throw new Error(JSON.parse(body).detail || `Error ${res.status}`);
        return JSON.parse(body);
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return (
    <Card title="Model Metrics">
      <div className={styles.error}>{error}</div>
    </Card>
  );

  if (!data) return (
    <Card title="Model Metrics">
      <div className={styles.loading}>Loading metrics...</div>
    </Card>
  );

  // Bar chart data — one entry per metric, one bar per model
  const barData = METRIC_KEYS.map((key) => {
    const entry = { metric: key.replace("_", " ").toUpperCase() };
    data.models.forEach((m) => { entry[m.model] = +(m[key] * 100).toFixed(1); });
    return entry;
  });

  // Radar data — one entry per model
  const radarData = METRIC_KEYS.map((key) => {
    const entry = { metric: key.replace("_", " ").toUpperCase() };
    data.models.forEach((m) => { entry[m.model] = +(m[key] * 100).toFixed(1); });
    return entry;
  });

  return (
    <div className={styles.page}>
      <Card title="Model Comparison">
        <div className={styles.bestBadge}>
          Best model: <span className={styles.badge}>{data.best_model}</span>
        </div>

        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Model</th>
                {METRIC_KEYS.map((k) => <th key={k}>{k.replace("_", " ").toUpperCase()}</th>)}
              </tr>
            </thead>
            <tbody>
              {data.models.map((m) => (
                <tr key={m.model} className={m.model === data.best_model ? styles.best : ""}>
                  <td>
                    <span className={styles.dot} style={{ background: COLORS[m.model] || "#888" }} />
                    {m.model}
                    {m.model === data.best_model && <span className={styles.crown}>★</span>}
                  </td>
                  {METRIC_KEYS.map((k) => (
                    <td key={k}>
                      <div className={styles.metricCell}>
                        <div className={styles.metricBar} style={{ width: `${m[k] * 100}%`, background: COLORS[m.model] || "#888" }} />
                        <span>{(m[k] * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className={styles.charts}>
        <Card title="Metrics by Model (Bar)">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Legend />
              {data.models.map((m) => (
                <Bar key={m.model} dataKey={m.model} fill={COLORS[m.model] || "#888"} radius={[3, 3, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Model Profile (Radar)">
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `${v}%`} />
              {data.models.map((m) => (
                <Radar
                  key={m.model}
                  name={m.model}
                  dataKey={m.model}
                  stroke={COLORS[m.model] || "#888"}
                  fill={COLORS[m.model] || "#888"}
                  fillOpacity={0.15}
                />
              ))}
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}

import { useState, useEffect } from "react";
import Card from "../components/Card";
import Button from "../components/Button";
import styles from "./Health.module.css";

export default function Health() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastChecked, setLastChecked] = useState(null);

  const check = async () => {
    setLoading(true);
    try {
      const res = await fetch("/health");
      setData({ ok: res.ok, ...(await res.json()) });
    } catch {
      setData({ ok: false, status: "unreachable", model_loaded: false });
    } finally {
      setLoading(false);
      setLastChecked(new Date().toLocaleTimeString());
    }
  };

  useEffect(() => { check(); }, []);

  return (
    <div className={styles.page}>
      <Card title="API Health Status">
        <div className={styles.grid}>
          <StatusTile
            label="API Status"
            value={data ? (data.ok ? "Online" : "Offline") : "—"}
            ok={data?.ok}
          />
          <StatusTile
            label="Model Loaded"
            value={data ? (data.model_loaded ? "Yes" : "No") : "—"}
            ok={data?.model_loaded}
          />
          <StatusTile
            label="Endpoint"
            value="localhost:8000"
            ok={data?.ok}
          />
        </div>

        <div className={styles.footer}>
          {lastChecked && <span className={styles.ts}>Last checked: {lastChecked}</span>}
          <Button onClick={check} loading={loading} variant="ghost">Refresh</Button>
        </div>
      </Card>

      <Card title="Available Endpoints">
        <table className={styles.table}>
          <thead>
            <tr><th>Method</th><th>Path</th><th>Description</th></tr>
          </thead>
          <tbody>
            {[
              ["GET",  "/health",           "Health check"],
              ["POST", "/predict/booking",  "Predict booking completion"],
              ["POST", "/sentiment/analyse","Analyse review sentiment"],
              ["GET",  "/metrics/",         "Model evaluation metrics"],
              ["GET",  "/docs",             "Swagger UI"],
            ].map(([method, path, desc]) => (
              <tr key={path}>
                <td><span className={`${styles.method} ${styles[method.toLowerCase()]}`}>{method}</span></td>
                <td><code>{path}</code></td>
                <td>{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}

function StatusTile({ label, value, ok }) {
  return (
    <div className={`${styles.tile} ${ok === undefined ? "" : ok ? styles.tileOk : styles.tileBad}`}>
      <div className={styles.tileLabel}>{label}</div>
      <div className={styles.tileValue}>{value}</div>
    </div>
  );
}

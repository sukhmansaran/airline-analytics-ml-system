import { useState } from "react";
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from "recharts";
import Card from "../components/Card";
import Button from "../components/Button";
import styles from "./Sentiment.module.css";

const BADGE = { Positive: styles.pos, Negative: styles.neg, Neutral: styles.neu };

export default function Sentiment() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/sentiment/analyse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const body = await res.text();
      if (!res.ok) {
        let detail = `API error (${res.status})`;
        try { detail = JSON.parse(body).detail || detail; } catch {}
        throw new Error(detail);
      }
      setResult(JSON.parse(body));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const radarData = result
    ? [
        { label: "Negative", VADER: +(result.vader_neg * 100).toFixed(1), RoBERTa: +(result.roberta_neg * 100).toFixed(1) },
        { label: "Neutral",  VADER: +(result.vader_neu * 100).toFixed(1), RoBERTa: +(result.roberta_neu * 100).toFixed(1) },
        { label: "Positive", VADER: +(result.vader_pos * 100).toFixed(1), RoBERTa: +(result.roberta_pos * 100).toFixed(1) },
      ]
    : [];

  return (
    <div className={styles.page}>
      <Card title="Review Sentiment Analyser">
        <form onSubmit={handleSubmit} className={styles.form}>
          <textarea
            rows={5}
            placeholder="Paste a British Airways review here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            className={styles.textarea}
          />
          <div className={styles.actions}>
            <Button type="submit" loading={loading} disabled={!text.trim()}>Analyse</Button>
            {result && <Button type="button" variant="ghost" onClick={() => { setResult(null); setText(""); }}>Clear</Button>}
          </div>
        </form>

        {error && <div className={styles.error}>{error}</div>}

        {result && (
          <div className={styles.results}>
            <div className={styles.verdict}>
              Overall Sentiment:
              <span className={`${styles.badge} ${BADGE[result.pred_result]}`}>{result.pred_result}</span>
            </div>

            <div className={styles.scores}>
              {[
                { label: "VADER Neg",    val: result.vader_neg },
                { label: "VADER Neu",    val: result.vader_neu },
                { label: "VADER Pos",    val: result.vader_pos },
                { label: "RoBERTa Neg",  val: result.roberta_neg },
                { label: "RoBERTa Neu",  val: result.roberta_neu },
                { label: "RoBERTa Pos",  val: result.roberta_pos },
              ].map(({ label, val }) => (
                <div key={label} className={styles.scoreRow}>
                  <span className={styles.scoreLabel}>{label}</span>
                  <div className={styles.bar}>
                    <div className={styles.fill} style={{ width: `${(val * 100).toFixed(1)}%` }} />
                  </div>
                  <span className={styles.scoreVal}>{(val * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>

            <div className={styles.chart}>
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="label" />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Radar name="VADER"   dataKey="VADER"   stroke="#003580" fill="#003580" fillOpacity={0.25} />
                  <Radar name="RoBERTa" dataKey="RoBERTa" stroke="#c8102e" fill="#c8102e" fillOpacity={0.25} />
                </RadarChart>
              </ResponsiveContainer>
              <div className={styles.legend}>
                <span className={styles.legendBlue}>■ VADER</span>
                <span className={styles.legendRed}>■ RoBERTa</span>
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

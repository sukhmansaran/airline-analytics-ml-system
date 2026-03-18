import { useState } from "react";
import Card from "../components/Card";
import FormField from "../components/FormField";
import Button from "../components/Button";
import styles from "./Predict.module.css";

const DEFAULTS = {
  num_passengers: 1,
  sales_channel: "Internet",
  trip_type: "RoundTrip",
  purchase_lead: 30,
  length_of_stay: 7,
  flight_hour: 10,
  flight_day: "Mon",
  route: "LHRBKK",
  booking_origin: "United Kingdom",
  wants_extra_baggage: 0,
  wants_preferred_seat: 0,
  wants_in_flight_meals: 0,
  flight_duration: 11.5,
};

export default function Predict() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        ...form,
        num_passengers: Number(form.num_passengers),
        purchase_lead: Number(form.purchase_lead),
        length_of_stay: Number(form.length_of_stay),
        flight_hour: Number(form.flight_hour),
        wants_extra_baggage: Number(form.wants_extra_baggage),
        wants_preferred_seat: Number(form.wants_preferred_seat),
        wants_in_flight_meals: Number(form.wants_in_flight_meals),
        flight_duration: Number(form.flight_duration),
      };
      const res = await fetch("/predict/booking", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
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

  return (
    <div className={styles.page}>
      <Card title="Booking Completion Predictor">
        <form onSubmit={handleSubmit} className={styles.grid}>
          <FormField label="Passengers">
            <input type="number" min={1} value={form.num_passengers} onChange={(e) => set("num_passengers", e.target.value)} />
          </FormField>
          <FormField label="Sales Channel">
            <select value={form.sales_channel} onChange={(e) => set("sales_channel", e.target.value)}>
              <option>Internet</option>
              <option>Mobile</option>
            </select>
          </FormField>
          <FormField label="Trip Type">
            <select value={form.trip_type} onChange={(e) => set("trip_type", e.target.value)}>
              <option>RoundTrip</option>
              <option>OneWay</option>
              <option>CircleTrip</option>
            </select>
          </FormField>
          <FormField label="Purchase Lead (days)">
            <input type="number" min={0} value={form.purchase_lead} onChange={(e) => set("purchase_lead", e.target.value)} />
          </FormField>
          <FormField label="Length of Stay (days)">
            <input type="number" min={0} value={form.length_of_stay} onChange={(e) => set("length_of_stay", e.target.value)} />
          </FormField>
          <FormField label="Flight Hour">
            <input type="number" min={0} max={23} value={form.flight_hour} onChange={(e) => set("flight_hour", e.target.value)} />
          </FormField>
          <FormField label="Flight Day">
            <select value={form.flight_day} onChange={(e) => set("flight_day", e.target.value)}>
              {["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].map((d) => <option key={d}>{d}</option>)}
            </select>
          </FormField>
          <FormField label="Route">
            <input type="text" value={form.route} onChange={(e) => set("route", e.target.value)} />
          </FormField>
          <FormField label="Booking Origin">
            <input type="text" value={form.booking_origin} onChange={(e) => set("booking_origin", e.target.value)} />
          </FormField>
          <FormField label="Flight Duration (hrs)">
            <input type="number" step="0.1" min={0} value={form.flight_duration} onChange={(e) => set("flight_duration", e.target.value)} />
          </FormField>
          <FormField label="Extra Baggage">
            <select value={form.wants_extra_baggage} onChange={(e) => set("wants_extra_baggage", e.target.value)}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </FormField>
          <FormField label="Preferred Seat">
            <select value={form.wants_preferred_seat} onChange={(e) => set("wants_preferred_seat", e.target.value)}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </FormField>
          <FormField label="In-flight Meals">
            <select value={form.wants_in_flight_meals} onChange={(e) => set("wants_in_flight_meals", e.target.value)}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </FormField>

          <div className={styles.submit}>
            <Button type="submit" loading={loading}>Predict</Button>
          </div>
        </form>

        {error && <div className={styles.error}>{error}</div>}

        {result && (
          <div className={`${styles.result} ${result.booking_complete ? styles.yes : styles.no}`}>
            <div className={styles.verdict}>
              {result.booking_complete ? "✅ Likely to Complete Booking" : "❌ Unlikely to Complete Booking"}
            </div>
            <div className={styles.meta}>
              Confidence: <strong>{(result.probability * 100).toFixed(1)}%</strong>
              &nbsp;·&nbsp; Model: <strong>{result.model_used}</strong>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

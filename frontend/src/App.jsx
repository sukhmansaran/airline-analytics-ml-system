import { Routes, Route, NavLink } from "react-router-dom";
import Predict from "./pages/Predict";
import Sentiment from "./pages/Sentiment";
import Metrics from "./pages/Metrics";
import Health from "./pages/Health";
import styles from "./App.module.css";

export default function App() {
  return (
    <div className={styles.layout}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <span className={styles.logo}>✈</span>
          <span>British Airways DS</span>
        </div>
        <nav className={styles.nav}>
          <NavLink to="/" end className={({ isActive }) => isActive ? styles.active : ""}>Booking Predictor</NavLink>
          <NavLink to="/sentiment" className={({ isActive }) => isActive ? styles.active : ""}>Sentiment Analyser</NavLink>
          <NavLink to="/metrics" className={({ isActive }) => isActive ? styles.active : ""}>Model Metrics</NavLink>
          <NavLink to="/health" className={({ isActive }) => isActive ? styles.active : ""}>API Health</NavLink>
        </nav>
      </header>

      <main className={styles.main}>
        <Routes>
          <Route path="/" element={<Predict />} />
          <Route path="/sentiment" element={<Sentiment />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/health" element={<Health />} />
        </Routes>
      </main>

      <footer className={styles.footer}>
        British Airways Data Science · FastAPI + React
      </footer>
    </div>
  );
}

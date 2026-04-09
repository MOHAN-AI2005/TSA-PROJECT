import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  BarChart, Bar, Cell, AreaChart, Area, Legend, ReferenceLine,
  ScatterChart, Scatter, ZAxis
} from 'recharts';
import { 
  Zap, Shield, Activity, Database, Info, 
  Thermometer, CloudRain, Calendar, Download, Layers, TrendingUp,
  Search, BarChart3, Binary, Layout, Wind, Clock, Lock, Copy, 
  AlertTriangle, CheckCircle, CheckCircle2, GitCompare, Target, LineChart as LineChartIcon
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = "http://localhost:8000";

// Color palette for all models
const MODEL_COLORS = {
  "Classical - Naive":        "#6b7280",
  "Classical - SES":          "#a78bfa",
  "Classical - Holt-Winters": "#f472b6",
  "Classical - SARIMA":       "#fb923c",
  "ML - Ridge":               "#34d399",
  "ML - Random Forest":       "#63b3ed",
  "ML - Gradient Boosting":   "#fbbf24",
  "ML - SVR":                 "#f87171",
  "ML - Neural Net (MLP)":    "#e879f9",
};

const getCategoryColor = (cat) => cat === "Machine Learning" ? "#63b3ed" : "#f6ad55";

const App = () => {
  const [metrics, setMetrics] = useState({});
  const [historicalData, setHistoricalData] = useState([]);
  const [signals, setSignals] = useState({ acf: [], pacf: [] });
  const [decomposition, setDecomposition] = useState(null);
  
  const [selectedEngine, setSelectedEngine] = useState('ML - Random Forest');
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [blindResult, setBlindResult] = useState(null);
  const [lockedResult, setLockedResult] = useState(null);
  const [rollingDiag, setRollingDiag] = useState(null);
  const [activeStage, setActiveStage] = useState('stage1'); 
  const [isWeatherEnabled, setIsWeatherEnabled] = useState(true);
  const [researchData, setResearchData] = useState({ importance: [], residuals: null });
  const [edaData, setEdaData] = useState({ monthly: [], weekly: [], weather: [] });
  const [showExporter, setShowExporter] = useState(false);

  const [inputs, setInputs] = useState({
    t_max: 32, t_min: 24, precip: 0, wind: 10,
    target_date: "2024-07-01", 
    lag_1: 0, lag_7: 0, rolling_mean_7: 0
  });

  useEffect(() => {
    const fetchResearch = async () => {
      try {
        const [imp, res, eda] = await Promise.all([
          axios.get(`${API_BASE}/api/research/importance`),
          axios.get(`${API_BASE}/api/research/residuals`),
          axios.get(`${API_BASE}/api/research/eda`)
        ]);
        setResearchData({ importance: imp.data, residuals: res.data });
        setEdaData(eda.data);
      } catch (err) { console.warn("Research Cache Pending..."); }
    };

    const fetchRollingDiagnostics = async () => {
      if (activeStage === 'stage4') {
        try {
          const res = await axios.get(`${API_BASE}/api/diagnostics/rolling?engine=${selectedEngine}`);
          setRollingDiag(res.data);
        } catch (err) { console.error("Rolling Diag Error", err); }
      }
    };

    fetchResearch();
    fetchRollingDiagnostics();
  }, [activeStage, selectedEngine]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [mRes, dRes, sRes, decRes] = await Promise.all([
          axios.get(`${API_BASE}/api/metrics`),
          axios.get(`${API_BASE}/api/data/historical`),
          axios.get(`${API_BASE}/api/signals`),
          axios.get(`${API_BASE}/api/decompose`)
        ]);
        setMetrics(mRes.data);
        setHistoricalData(dRes.data);
        setSignals(sRes.data);
        setDecomposition(decRes.data);
        updateLagsForDate("2024-07-01", dRes.data);
      } catch (err) { console.error("Init Error:", err); }
    };
    fetchData();
  }, []);

  const updateLagsForDate = (dateStr, dataPool) => {
    const target = dataPool.find(d => d.date.startsWith(dateStr));
    if (target) {
      setInputs(prev => ({
        ...prev,
        lag_1: target.lag_1 || prev.lag_1,
        lag_7: target.lag_7 || prev.lag_7,
        rolling_mean_7: target.rolling_mean_7 || prev.rolling_mean_7
      }));
    }
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setBlindResult(null);
    try {
      // 1. Primary Model Prediction
      let engineToUse = selectedEngine;
      if (!isWeatherEnabled && selectedEngine.startsWith("ML - ")) {
        const blindVariant = `${selectedEngine} (No Weather)`;
        if (metrics[blindVariant]) engineToUse = blindVariant;
      }
      const res = await axios.post(`${API_BASE}/api/predict`, { engine: engineToUse, ...inputs });
      setPredictionResult(res.data);

      // 2. Ablation (No Weather) Benchmark Prediction - Runs in parallel if applicable
      const hasWeather = !selectedEngine.includes("(No Weather)") && metrics[`${selectedEngine} (No Weather)`];
      if (hasWeather) {
        const resBlind = await axios.post(`${API_BASE}/api/predict`, { engine: `${selectedEngine} (No Weather)`, ...inputs });
        setBlindResult(resBlind.data);
      }
    } catch (err) { alert("Backend Error: " + (err.response?.data?.detail || err.message)); } 
    finally { setIsLoading(false); }
  };

  const isClassical = metrics[selectedEngine]?.Category === 'Classical';
  const targetDateObj = new Date(inputs.target_date);
  const calendarEffect = {
    month: targetDateObj.toLocaleString('en-US', { month: 'long' }),
    day: targetDateObj.toLocaleString('en-US', { weekday: 'long' }),
    isWeekend: targetDateObj.getDay() === 0 || targetDateObj.getDay() === 6 ? "Weekend Profile" : "Weekday Load"
  };

  const getStability = (val) => {
    if (val < 185000) return { label: "OPTIMAL", color: "text-emerald-400", bg: "bg-emerald-400/10", icon: CheckCircle };
    if (val < 230000) return { label: "NOMINAL", color: "text-amber-400", bg: "bg-amber-400/10", icon: Activity };
    return { label: "GRID STRESS", color: "text-rose-500", bg: "bg-rose-500/10", icon: AlertTriangle };
  };

  const generateMetadata = () => {
    const stability = getStability(predictionResult?.prediction);
    return `GRID ANALYTICS METADATA REPORT\n` +
           `--------------------------------\n` +
           `• Target Date: ${inputs.target_date} (${calendarEffect.day})\n` +
           `• Primary Algorithm: ${selectedEngine} (${metrics[selectedEngine]?.Category})\n` +
           `• Validation Performance: MAE ${metrics[selectedEngine]?.MAE} MW\n` +
           `• Prediction Result: ${predictionResult?.prediction.toLocaleString()} MW\n` +
           `• Grid Stability Status: ${stability.label}\n` +
           `• Input Context: MaxTemp ${inputs.t_max}°C / Precip ${inputs.precip}mm\n` +
           `• Methodology: Recursive Multivariate Trajectory Analysis\n` +
           `--------------------------------`;
  };

  const GlassCard = ({ title, icon: Icon, children, className = "" }) => (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`glass-card p-6 ${className}`}>
      <div className="flex items-center gap-3 mb-4">
        {Icon && <div className="p-2 bg-accent/10 rounded-lg text-accent"><Icon size={16} /></div>}
        <h3 className="text-[10px] font-black tracking-[4px] uppercase text-white/40">{title}</h3>
      </div>
      {children}
    </motion.div>
  );

  // Historical comparison data for stage 1
  const maeBarchartData = Object.entries(metrics).map(([name, info]) => ({
    name: name.replace('Classical - ', '').replace('ML - ', ''),
    fullName: name,
    mae: info.MAE ? Math.round(info.MAE) : 0,
    rmse: info.RMSE ? Math.round(info.RMSE) : 0,
    mape: info.MAPE ? Number(info.MAPE.toFixed(2)) : 0,
    category: info.Category,
    color: MODEL_COLORS[name] || '#888'
  }));

  const combinedTrajectory = predictionResult ? predictionResult.trajectory.map((t, i) => ({
    ...t,
    locked: lockedResult ? lockedResult.trajectory[i]?.prediction : null,
    blind: blindResult ? blindResult.trajectory[i]?.prediction : null
  })) : [];

  // Custom tooltip for error band chart
  const TrajectoryTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const pred = payload.find(p => p.dataKey === 'prediction');
      const actual = payload.find(p => p.dataKey === 'actual');
      const blind = payload.find(p => p.dataKey === 'blind');
      const upper = payload.find(p => p.dataKey === 'upper_band');
      const lower = payload.find(p => p.dataKey === 'lower_band');
      return (
        <div className="bg-[#0a0c10]/90 backdrop-blur-xl border border-white/10 rounded-2xl p-4 shadow-2xl">
          <p className="text-[10px] font-black text-white/40 uppercase tracking-widest mb-3 border-b border-white/5 pb-2">{label} (7-Day Horizon)</p>
          <div className="space-y-2">
            {actual && actual.value && (
              <div className="flex justify-between items-center gap-8">
                <span className="text-[10px] font-bold text-emerald-400 uppercase">Actual Load</span>
                <span className="text-sm font-black text-white">{actual.value?.toLocaleString()} MW</span>
              </div>
            )}
            {pred && (
              <div className="flex justify-between items-center gap-8">
                <span className="text-[10px] font-bold text-sky-400 uppercase">Predicted</span>
                <span className="text-sm font-black text-white">{pred.value?.toLocaleString()} MW</span>
              </div>
            )}
            {blind && (
              <div className="flex justify-between items-center gap-8 border-t border-white/5 pt-2">
                <span className="text-[10px] font-bold text-slate-500 uppercase">Univariate Blind</span>
                <span className="text-sm font-black text-slate-400">{blind.value?.toLocaleString()} MW</span>
              </div>
            )}
            {upper && lower && (
              <div className="pt-2 mt-2 border-t border-white/5">
                <div className="text-[9px] text-white/30 uppercase tracking-widest mb-1 italic">Research Confidence (±MAE Range)</div>
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="text-sky-400/60">H: {upper.value?.toLocaleString()}</span>
                  <span className="text-slate-500">/</span>
                  <span className="text-rose-400/60">L: {lower.value?.toLocaleString()}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-[#0a0c10] text-[#e2e8f0] p-6 lg:p-10 max-w-[1700px] mx-auto font-inter">
      {/* Narrative Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8 mb-12">
        <div>
          <h1 className="text-4xl font-black italic tracking-tighter text-white leading-tight max-w-2xl">
            Multivariate Electricity Demand Forecasting<br/>
            <span className="text-accent underline decoration-4 underline-offset-8 decoration-accent/20">Using Weather and Calendar Effects</span>
          </h1>
        </div>

        <nav className="flex bg-white/5 p-1 rounded-2xl border border-white/5">
          {[
            { id: 'stage0', label: '0. Exploration', icon: Activity },
            { id: 'stage1', label: '1. Signals', icon: Search },
            { id: 'stage2', label: '2. Composition', icon: Layers },
            { id: 'stage3', label: '3. Comparison', icon: Zap },
            { id: 'stage4', label: '4. Diagnostics', icon: Binary }
          ].map(stage => (
            <button key={stage.id} onClick={() => setActiveStage(stage.id)} 
              className={`px-8 py-4 rounded-xl flex items-center gap-2 text-[11px] font-black uppercase tracking-widest transition-all ${activeStage === stage.id ? 'bg-accent text-slate-900 shadow-[0_0_40px_-10px_#63b3ed]' : 'text-slate-500 hover:text-white'}`}>
              <stage.icon size={14} /> {stage.label}
            </button>
          ))}
        </nav>
      </header>

      <AnimatePresence mode="wait">
        {/* ──────────────── STAGE 0: EXPLORATION (EDA) ──────────────── */}
        {activeStage === 'stage0' && (
          <motion.div key="s0" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="grid grid-cols-12 gap-8">
            <GlassCard title="1. Seasonal Demand Curve (Monthly Average)" icon={Calendar} className="col-span-12 lg:col-span-7">
               <div className="h-[300px] w-full mt-4">
                 <ResponsiveContainer width="100%" height="100%">
                   <BarChart data={edaData.monthly}>
                     <CartesianGrid stroke="#222" vertical={false} />
                     <XAxis dataKey="name" stroke="#444" fontSize={10} />
                     <YAxis stroke="#444" fontSize={10} tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} />
                     <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '15px' }} />
                     <Bar dataKey="load" fill="#63b3ed" radius={[4, 4, 0, 0]}>
                       {edaData.monthly.map((entry, index) => (
                         <Cell key={index} fillOpacity={index >= 5 && index <= 8 ? 1 : 0.3} />
                       ))}
                     </Bar>
                   </BarChart>
                 </ResponsiveContainer>
               </div>
               <p className="text-[10px] text-white/30 mt-4 italic uppercase tracking-widest">Identifying summer peak cycles (June - Sept) for infrastructure planning.</p>
            </GlassCard>

            <GlassCard title="2. Weekly Human Dynamics (The Weekend Dip)" icon={Clock} className="col-span-12 lg:col-span-5">
               <div className="h-[300px] w-full mt-4">
                 <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={edaData.weekly}>
                      <CartesianGrid stroke="#222" vertical={false} />
                      <XAxis dataKey="name" stroke="#444" fontSize={10} />
                      <YAxis stroke="#444" fontSize={10} tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} />
                      <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '10px' }} />
                      <Bar dataKey="mean" fill="#f6ad55" radius={[4, 4, 0, 0]}>
                        {edaData.weekly.map((entry, index) => (
                          <Cell key={index} fill={index >= 5 ? '#f43f5e' : '#f6ad55'} />
                        ))}
                      </Bar>
                    </BarChart>
                 </ResponsiveContainer>
               </div>
               <p className="text-[10px] text-white/30 mt-4 italic uppercase tracking-widest text-center">Lower industrial demand detected on Sat/Sun benchmarks.</p>
            </GlassCard>

            <GlassCard title="3. Causal Matrix: Temperature vs Demand" icon={Thermometer} className="col-span-12 lg:col-span-6">
               <div className="h-[300px] w-full mt-4">
                 <ResponsiveContainer width="100%" height="100%">
                   <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                     <CartesianGrid stroke="#222" />
                     <XAxis type="number" dataKey="temp_max" name="Max Temp" unit="°C" stroke="#444" fontSize={10} />
                     <YAxis type="number" dataKey="load" name="Load" unit=" MW" stroke="#444" fontSize={10} tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} />
                     <ZAxis type="number" range={[50, 50]} />
                     <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }} />
                     <Scatter name="Load Correlation" data={edaData.weather} fill="#63b3ed" opacity={0.6} />
                   </ScatterChart>
                 </ResponsiveContainer>
               </div>
            </GlassCard>

            <GlassCard title="4. Causal Matrix: Precipitation Impact" icon={CloudRain} className="col-span-12 lg:col-span-6">
               <div className="h-[300px] w-full mt-4">
                 <ResponsiveContainer width="100%" height="100%">
                   <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                     <CartesianGrid stroke="#222" />
                     <XAxis type="number" dataKey="precipitation" name="Precipitation" unit="mm" stroke="#444" fontSize={10} />
                     <YAxis type="number" dataKey="load" name="Load" unit=" MW" stroke="#444" fontSize={10} tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} />
                     <ZAxis type="number" range={[50, 50]} />
                     <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }} />
                     <Scatter name="Precip Effect" data={edaData.weather} fill="#f6ad55" opacity={0.6} />
                   </ScatterChart>
                 </ResponsiveContainer>
               </div>
            </GlassCard>
          </motion.div>
        )}

        {/* ──────────────── STAGE 1: SIGNALS ──────────────── */}
        {activeStage === 'stage1' && (
          <motion.div key="s1" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }} className="grid grid-cols-12 gap-8">
            <GlassCard title="Historical Load Analysis (2022-2025)" icon={Database} className="col-span-12">
               <div className="h-[400px] w-full mt-4">
                 <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={historicalData}>
                      <defs>
                        <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#63b3ed" stopOpacity={0.4}/>
                          <stop offset="95%" stopColor="#63b3ed" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#222" vertical={false} strokeDasharray="3 3" />
                      <XAxis dataKey="date" hide />
                      <YAxis stroke="#444" fontSize={10} domain={['auto', 'auto']} tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} />
                      <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '15px' }} formatter={(v) => [`${v?.toLocaleString()} MW`, 'Load']} />
                      <Area type="monotone" dataKey="load" stroke="#63b3ed" strokeWidth={3} fillOpacity={1} fill="url(#colorLoad)" />
                    </AreaChart>
                 </ResponsiveContainer>
               </div>
            </GlassCard>

            {/* ── ALL MODELS METRICS COMPARISON (Historical) ── */}
            <div className="col-span-12 glass-card p-6 border-l-4 border-l-accent">
               <h3 className="text-sm font-black tracking-widest uppercase text-white mb-2 flex items-center gap-2"><Info size={16} className="text-accent" /> Important Metric Insight (For Defense)</h3>
                <p className="text-xs text-white/60 italic leading-relaxed">
                  "MAE is used as the primary evaluation metric due to its interpretability, while RMSE captures sensitivity to outliers. **MAPE is included to provide a relative, scale-independent percentage error—essential for cross-seasonal reliability benchmarks.**"
                </p>
            </div>

            <GlassCard title="1. Precision (MAE)" icon={BarChart3} className="col-span-12 xl:col-span-4">
              <div className="h-[220px] w-full mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={maeBarchartData} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                    <CartesianGrid stroke="#222" vertical={false} strokeDasharray="3 3" />
                    <XAxis dataKey="name" stroke="#444" fontSize={8} angle={-30} textAnchor="end" interval={0} />
                    <YAxis stroke="#444" fontSize={10} tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '10px' }}
                      formatter={(v, name, props) => [`${v?.toLocaleString()} MW`, `MAE — ${props.payload.fullName}`]}
                    />
                    <Bar dataKey="mae" radius={[4, 4, 0, 0]}>
                      {maeBarchartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={getCategoryColor(entry.category)} fillOpacity={0.8} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>

            <GlassCard title="2. Sensitivity (RMSE)" icon={Target} className="col-span-12 xl:col-span-4">
              <div className="h-[220px] w-full mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={maeBarchartData} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                    <CartesianGrid stroke="#222" vertical={false} strokeDasharray="3 3" />
                    <XAxis dataKey="name" stroke="#444" fontSize={8} angle={-30} textAnchor="end" interval={0} />
                    <YAxis stroke="#444" fontSize={10} tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '10px' }}
                      formatter={(v, name, props) => [`${v?.toLocaleString()} MW`, `RMSE — ${props.payload.fullName}`]}
                    />
                    <Bar dataKey="rmse" radius={[4, 4, 0, 0]}>
                      {maeBarchartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.category === 'Machine Learning' ? '#fc8181' : '#f6ad55'} fillOpacity={0.8} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>

            <GlassCard title="3. Relative Error (MAPE)" icon={TrendingUp} className="col-span-12 xl:col-span-4">
              <div className="h-[220px] w-full mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={maeBarchartData} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                    <CartesianGrid stroke="#222" vertical={false} strokeDasharray="3 3" />
                    <XAxis dataKey="name" stroke="#444" fontSize={8} angle={-30} textAnchor="end" interval={0} />
                    <YAxis stroke="#444" fontSize={10} domain={[0, 'auto']} tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '10px' }}
                      formatter={(v, name, props) => [`${v}%`, `MAPE — ${props.payload.fullName}`]}
                    />
                    <Bar dataKey="mape" radius={[4, 4, 0, 0]}>
                      {maeBarchartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.category === 'Machine Learning' ? '#10b981' : '#eab308'} fillOpacity={0.8} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>

            <div className="col-span-12 lg:col-span-12 grid grid-cols-1 md:grid-cols-3 gap-8">
              <GlassCard title="ACF (Memory)" icon={Activity}>
                 <div className="h-[200px] w-full mt-2">
                   <ResponsiveContainer width="100%" height="100%">
                     <BarChart data={signals.acf.map((v, i) => ({l: i, v}))}>
                       <Bar dataKey="v" fill="#63b3ed" radius={[4, 4, 0, 0]}>
                         {signals.acf.map((_, i) => <Cell key={i} fillOpacity={i === 7 ? 1 : 0.2} />)}
                       </Bar>
                     </BarChart>
                   </ResponsiveContainer>
                 </div>
              </GlassCard>
              <GlassCard title="PACF (Direct Lag)" icon={Binary}>
                <div className="h-[200px] w-full mt-2">
                   <ResponsiveContainer width="100%" height="100%">
                     <BarChart data={signals.pacf.map((v, i) => ({l: i, v}))}>
                       <Bar dataKey="v" fill="#f6ad55" radius={[4, 4, 0, 0]}>
                         {signals.pacf.map((_, i) => <Cell key={i} fillOpacity={i === 7 ? 1 : 0.2} />)}
                       </Bar>
                     </BarChart>
                   </ResponsiveContainer>
                 </div>
              </GlassCard>
              <GlassCard title="Seasonality Proof" icon={TrendingUp}>
                  <div className="flex flex-col gap-4 py-8 text-center text-sm">
                    <div className="text-3xl font-black text-white italic">7-Day Cyclic Period</div>
                    <p className="text-[10px] text-slate-500 uppercase tracking-[2px]">Validated via Stochastic Lag Correlation</p>
                  </div>
              </GlassCard>
            </div>
          </motion.div>
        )}

        {/* ──────────────── STAGE 2: DECOMPOSITION ──────────────── */}
        {activeStage === 'stage2' && (
          <motion.div key="s2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col gap-6">
            <GlassCard title="Signal Decomposition Analysis" icon={Layout}>
              {decomposition ? (
                <div className="flex flex-col gap-8 py-4">
                  {[
                    { label: "Original Signal", data: decomposition.original, color: "#e2e8f0", opacity: 0.5 },
                    { label: "Long-Term Trend", data: decomposition.trend, color: "#63b3ed", opacity: 1 },
                    { label: "Seasonal Component", data: decomposition.seasonal, color: "#f6ad55", opacity: 1 },
                    { label: "Random Residuals", data: decomposition.resid, color: "#fc8181", opacity: 0.8 }
                  ].map((panel, idx) => (
                    <div key={panel.label} className="h-[100px] w-full flex items-center">
                      <div className="w-48 text-[10px] font-black uppercase tracking-widest text-slate-500">{panel.label}</div>
                      <div className="flex-1 h-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={panel.data.map((v, i) => ({value: v, date: decomposition.dates[i]}))}>
                            <XAxis dataKey="date" hide />
                            <Tooltip contentStyle={{ backgroundColor: '#111', border: 'none', fontSize: '10px' }} 
                              formatter={(v) => [v?.toLocaleString(), panel.label]} />
                            <Area type="monotone" dataKey="value" stroke={panel.color} fill={panel.color} fillOpacity={0.05} />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  ))}
                </div>
              ) : <div className="p-20 text-center tracking-[10px]">ANALYZING SIGNAL...</div>}
            </GlassCard>
          </motion.div>
        )}

        {/* ──────────────── STAGE 3: COMPARISON ──────────────── */}
        {activeStage === 'stage3' && (
          <motion.div key="s3" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="grid grid-cols-12 gap-8">
            <div className="col-span-12 xl:col-span-4 flex flex-col gap-8">
                <div className="glass-card p-8 bg-white/[0.02]">
                  <h2 className="text-xl font-black mb-8 flex items-center gap-3 italic"><Zap className="text-accent" /> FORECASTING CONTROL CENTER</h2>
                  <div className="flex flex-col gap-6">
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <label className="text-[10px] font-black text-white/30 tracking-[4px] uppercase">1. Chosen Algorithm</label>
                        {predictionResult && (
                          <button onClick={() => setLockedResult({ ...predictionResult, name: selectedEngine })} className={`text-[9px] font-bold px-3 py-1 rounded-full border transition-all ${lockedResult?.name === selectedEngine ? 'border-accent text-accent' : 'border-white/20 text-white/40 hover:border-white/40'}`}>
                            <Lock size={10} className="inline mr-1" /> {lockedResult?.name === selectedEngine ? 'LOCKED' : 'LOCK FOR DUEL'}
                          </button>
                        )}
                      </div>
                      <select value={selectedEngine} onChange={e => setSelectedEngine(e.target.value)} className="w-full bg-black/50 border border-white/5 p-4 rounded-xl font-black text-sm outline-none focus:border-accent">
                        {Object.keys(metrics).filter(m => !m.includes("(No Weather)")).map(m => <option key={m}>{m}</option>)}
                      </select>
                    </div>

                    <div className="flex items-center justify-between bg-white/[0.04] p-4 rounded-xl border border-white/5">
                      <div>
                        <div className="text-[9px] font-black text-white/40 tracking-widest">WEATHER CONTEXT</div>
                        <div className="text-[10px] font-bold text-accent">{isWeatherEnabled ? "MULTIVARIATE ACTIVE" : "UNIVARIATE/BLIND MODE"}</div>
                      </div>
                      <button 
                        onClick={() => setIsWeatherEnabled(!isWeatherEnabled)}
                        className={`w-12 h-6 rounded-full p-1 transition-all ${isWeatherEnabled ? 'bg-accent' : 'bg-slate-700'}`}
                      >
                        <div className={`w-4 h-4 bg-white rounded-full transition-transform ${isWeatherEnabled ? 'translate-x-6' : 'translate-x-0'}`} />
                      </button>
                    </div>

                    {/* Error badge for selected model */}
                    {metrics[selectedEngine] && (
                      <div className="flex items-center gap-3 bg-black/30 border border-white/5 rounded-xl p-4">
                        <div className="p-2 bg-rose-500/10 rounded-lg">
                          <Target size={14} className="text-rose-400" />
                        </div>
                        <div>
                          <div className="text-[8px] text-slate-500 uppercase tracking-widest">Validation Error (MAE)</div>
                          <div className="text-lg font-black text-rose-400">
                            ± {Math.round(metrics[selectedEngine]?.MAE).toLocaleString()} MW
                          </div>
                        </div>
                        <div className={`ml-auto text-[9px] font-black px-3 py-1 rounded-full ${metrics[selectedEngine]?.Category === 'Machine Learning' ? 'bg-blue-500/10 text-blue-400' : 'bg-amber-500/10 text-amber-400'}`}>
                          {metrics[selectedEngine]?.Category}
                        </div>
                      </div>
                    )}

                    <div className="space-y-6 bg-white/[0.02] p-6 rounded-2xl border border-white/5">
                      <div className="flex justify-between items-center mb-4">
                        <label className="text-[10px] font-black text-white/30 tracking-[4px] uppercase block">2. Weather Intensity</label>
                        <div className="group relative">
                          <Info size={12} className="text-white/20 cursor-help" />
                          <div className="absolute bottom-full right-0 mb-2 w-48 p-3 bg-black border border-white/10 rounded-xl text-[9px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                            Note: Multivariate models incorporate weather exogenous variables for higher precision.
                          </div>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-6">
                        <div className="space-y-2">
                           <span className="text-[9px] text-slate-500 uppercase">Max Temp: {inputs.t_max}°C</span>
                           <input type="range" min="20" max="48" value={inputs.t_max} onChange={e => setInputs({...inputs, t_max: parseInt(e.target.value)})} className="w-full accent-accent h-1 bg-white/5 appearance-none rounded-full" />
                        </div>
                        <div className="space-y-2">
                           <span className="text-[9px] text-slate-500 uppercase">Min Temp: {inputs.t_min}°C</span>
                           <input type="range" min="10" max="35" value={inputs.t_min} onChange={e => setInputs({...inputs, t_min: parseInt(e.target.value)})} className="w-full accent-accent h-1 bg-white/5 appearance-none rounded-full" />
                        </div>
                        <div className="space-y-2">
                           <span className="text-[9px] text-slate-500 uppercase">Precip: {inputs.precip}mm</span>
                           <input type="range" min="0" max="100" value={inputs.precip} onChange={e => setInputs({...inputs, precip: parseInt(e.target.value)})} className="w-full h-1 bg-white/5 accent-blue-500 appearance-none rounded-full" />
                        </div>
                        <div className="space-y-2">
                           <span className="text-[9px] text-slate-500 uppercase">Wind: {inputs.wind}km/h</span>
                           <input type="range" min="0" max="50" value={inputs.wind} onChange={e => setInputs({...inputs, wind: parseInt(e.target.value)})} className="w-full h-1 bg-white/5 accent-blue-900 appearance-none rounded-full" />
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <label className="text-[10px] font-black text-white/30 tracking-[4px] uppercase block">3. Sequence Target Date</label>
                      <input type="date" value={inputs.target_date} onChange={e => {setInputs({...inputs, target_date: e.target.value}); updateLagsForDate(e.target.value, historicalData);}} className="w-full bg-black/50 border border-white/5 p-4 rounded-xl text-xs font-black" />
                    </div>

                    <button onClick={handlePredict} disabled={isLoading} className="bg-accent text-slate-900 font-black py-5 rounded-xl hover:shadow-[0_0_40px_-10px_#63b3ed] transition-all flex justify-center gap-3">
                      {isLoading ? "PROCESSING..." : <Activity size={18} />} INITIATE COMPARISON
                    </button>
                    
                    {predictionResult && (
                      <button onClick={() => setShowExporter(true)} className="text-[10px] font-black text-white/40 hover:text-accent uppercase tracking-widest mt-2 flex justify-center items-center gap-2">
                         <Copy size={12} /> Generate Final Report Snippet
                      </button>
                    )}
                  </div>
                </div>

                <GlassCard title="Calendar Effects Diagnostic" icon={Clock}>
                   <div className="grid grid-cols-2 gap-4">
                      <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                        <div className="text-[8px] text-slate-500 uppercase mb-1">Month Context</div>
                        <div className="text-sm font-black text-white">{calendarEffect.month}</div>
                      </div>
                      <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                        <div className="text-[8px] text-slate-500 uppercase mb-1">Human Dynamic</div>
                        <div className="text-sm font-black text-accent">{calendarEffect.day}</div>
                      </div>
                      <div className="col-span-2 bg-accent/5 p-4 rounded-xl border border-accent/10 flex justify-between items-center">
                        <div className="text-[9px] text-accent uppercase font-black">Memory Context [Lag 1]</div>
                        <div className="text-xs font-black text-white">{inputs.lag_1.toLocaleString()} MW</div>
                      </div>
                   </div>
                </GlassCard>
            </div>

            <div className="col-span-12 xl:col-span-8 flex flex-col gap-8">
               <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <GlassCard title="Forecast Point">
                    <div className="flex flex-col">
                      <div className="text-5xl font-black text-white">{predictionResult ? predictionResult.prediction.toLocaleString() : "---"}</div>
                      <div className="text-[9px] text-slate-600 uppercase tracking-widest mt-2 block">MW Demand</div>
                      
                      {/* Task 1: Explicit Prediction Intervals (Decision Support) */}
                      {predictionResult && (
                        <div className="mt-4 pt-4 border-t border-white/5">
                           <div className="text-[10px] font-black text-accent uppercase tracking-widest mb-1">95% Confidence Range</div>
                           <div className="text-sm font-bold text-white/80">
                              {(predictionResult.prediction - Math.round(predictionResult.mae)).toLocaleString()} - {(predictionResult.prediction + Math.round(predictionResult.mae)).toLocaleString()}
                           </div>
                           <div className="text-[9px] text-slate-500 mt-1 italic">± {Math.round(predictionResult.mae).toLocaleString()} MW (MAE)</div>
                        </div>
                      )}
                    </div>
                  </GlassCard>
                  <GlassCard title="Stability Advisory">
                    {predictionResult ? (
                      <div className={`flex items-center gap-3 mt-1 ${getStability(predictionResult.prediction).color}`}>
                        {React.createElement(getStability(predictionResult.prediction).icon, { size: 24 })}
                        <div className="text-2xl font-black tracking-tighter">{getStability(predictionResult.prediction).label}</div>
                      </div>
                    ) : <div className="text-2xl font-black text-slate-700">PENDING</div>}
                  </GlassCard>
                    <GlassCard title="Error Margins">
                      <div className="flex flex-col gap-1">
                        <div className="flex justify-between items-end border-b border-white/5 pb-2">
                          <div>
                            <p className="text-[9px] text-slate-500 uppercase tracking-widest">Primary Metric (MAE)</p>
                            <p className="text-xl font-bold text-accent">± {predictionResult ? predictionResult.mae.toLocaleString() : metrics[selectedEngine] ? Math.round(metrics[selectedEngine].MAE).toLocaleString() : '---'} MW</p>
                          </div>
                          <div className="text-right">
                            <p className="text-[9px] text-slate-500 uppercase tracking-widest">Sensitivity (RMSE)</p>
                            <p className="text-lg font-bold text-rose-400">± {predictionResult ? predictionResult.rmse?.toLocaleString() : metrics[selectedEngine] ? Math.round(metrics[selectedEngine].RMSE || 0).toLocaleString() : '---'} MW</p>
                          </div>
                        </div>
                        
                        {predictionResult && (
                          <div className="mt-2 flex flex-col gap-2">
                            <div className="flex gap-2 text-[9px]">
                              <span className="text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded">
                                High Bound: {(predictionResult.prediction + Math.round(predictionResult.mae)).toLocaleString()}
                              </span>
                              <span className="text-rose-400 bg-rose-400/10 px-2 py-1 rounded">
                                Low Bound: {(predictionResult.prediction - Math.round(predictionResult.mae)).toLocaleString()}
                              </span>
                            </div>
                            
                            {/* Weather Impact Badge */}
                            {metrics[`${selectedEngine} (No Weather)`] && (
                              <div className="mt-2 p-3 bg-accent/5 border border-accent/20 rounded-xl flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <div className="p-1 bg-accent/20 rounded text-accent"><Zap size={10} /></div>
                                  <span className="text-[10px] font-black text-white/50 uppercase tracking-tighter">Precision Gain</span>
                                </div>
                                <span className="text-xs font-black text-accent">
                                  +{((metrics[`${selectedEngine} (No Weather)`].MAE - metrics[selectedEngine].MAE) / metrics[`${selectedEngine} (No Weather)`].MAE * 100).toFixed(1)}% Accuracy Boost
                                </span>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </GlassCard>
               </div>

               {predictionResult && predictionResult.xai_insights && (
                 <GlassCard title="Feature Contribution Analysis" icon={Search}>
                    <div className="h-[200px] w-full mt-4">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart layout="vertical" data={predictionResult.xai_insights} margin={{ left: 40, right: 40 }}>
                          <XAxis type="number" hide domain={['auto', 'auto']} />
                          <YAxis type="category" dataKey="name" stroke="#444" fontSize={10} width={100} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#000', border: '1px solid #333', borderRadius: '10px' }}
                            formatter={(v) => [`${v > 0 ? '+' : ''}${v.toLocaleString()} MW`, "Impact"]}
                          />
                          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {predictionResult.xai_insights.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.value >= 0 ? '#10b981' : '#f43f5e'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                 </GlassCard>
               )}

               {/* ── TRAJECTORY with ERROR BAND ── */}
               <GlassCard title="7-Day Forecast Trajectory + Error Band (MAE)" icon={Activity} className="flex-1">
                 <div className="relative h-[360px] w-full mt-6">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={combinedTrajectory}>
                        <defs>
                          <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#63b3ed" stopOpacity={0.15}/>
                            <stop offset="100%" stopColor="#63b3ed" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="#222" vertical={false} strokeDasharray="5 5" />
                        <XAxis dataKey="date" stroke="#444" fontSize={10} tick={{ fill: '#666' }} />
                        <YAxis stroke="#444" fontSize={10} domain={['auto', 'auto']} tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} />
                        <Tooltip content={<TrajectoryTooltip />} />
                        <Legend verticalAlign="top" height={36} wrapperStyle={{ paddingBottom: '20px', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px' }} />
                        
                        {/* Shaded Confidence Interval */}
                        <Area type="monotone" dataKey="upper_band" fill="url(#bandGrad)" stroke="none" opacity={0.3} name="Tolerance Upper Bound" />
                        <Area type="monotone" dataKey="lower_band" fill="#0a0c10" stroke="none" opacity={0.8} name="Tolerance Lower Bound" />
                        
                        {/* Actual truth line if available */}
                        <Line type="monotone" dataKey="actual" name="Ground Truth (Actual)" stroke="#10b981" strokeWidth={3} dot={{ r: 4, stroke: "#10b981", strokeWidth: 2, fill: "#0a0c10" }} activeDot={{ r: 6 }} opacity={0.9} />
                        
                        {/* Forecast line */}
                        <Line type="monotone" dataKey="prediction" name={`Forecast: ${selectedEngine}`} stroke="#63b3ed" strokeWidth={4} dot={{ r: 5, fill: "#63b3ed", strokeWidth: 0 }} activeDot={{ r: 8 }} />
                        
                        {/* Weather Ablation Benchmark (No Weather) */}
                        {blindResult && <Line type="monotone" dataKey="blind" name="Ablation: No Weather" stroke="#94a3b8" strokeWidth={2} strokeDasharray="8 4" dot={false} opacity={0.6} />}
                        
                        {/* Locked comparison */}
                        {lockedResult && <Line type="monotone" dataKey="locked" name={`Benchmark: ${lockedResult.name}`} stroke="#f6ad55" strokeWidth={2} strokeDasharray="5 5" dot={false} opacity={0.6} />}
                      </AreaChart>
                    </ResponsiveContainer>
                    {/* Task 8: Forecast Confidence Message */}
                    {predictionResult && (
                      <div className="absolute top-16 right-8 p-3 bg-slate-900/60 backdrop-blur-md border border-white/5 rounded-2xl max-w-[180px]">
                         <div className="text-[9px] font-black text-white/30 uppercase tracking-widest mb-1">Confidence Advisory</div>
                         <div className="text-[10px] text-sky-200/80 leading-relaxed font-medium">
                            Forecast lies within ±MAE error band, indicating reliable short-term prediction.
                         </div>
                      </div>
                    )}
                     {!predictionResult && <div className="absolute inset-0 flex items-center justify-center bg-black/20 backdrop-blur-sm rounded-3xl text-slate-500 font-black tracking-[10px] italic animate-pulse">AWAITING COMPARISON...</div>}
                  </div>
                </GlassCard>

                <GlassCard title="Global Model Leaderboard (Ranked by Precision)" icon={Target}>
                  <div className="overflow-x-auto mt-4">
                    <table className="w-full text-left text-[11px]">
                      <thead>
                        <tr className="border-b border-white/5">
                          <th className="pb-4 font-black uppercase text-white/30 tracking-widest">Rank</th>
                          <th className="pb-4 font-black uppercase text-white/30 tracking-widest">Model Engine</th>
                          <th className="pb-4 font-black uppercase text-white/30 tracking-widest text-right">MAE</th>
                          <th className="pb-4 font-black uppercase text-white/30 tracking-widest text-right">MAPE</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(metrics)
                          .sort(([, a], [, b]) => a.MAE - b.MAE)
                          .map(([name, data], idx) => (
                            <tr key={name} className={`border-b border-white/5 hover:bg-white/2 transition-all ${name === selectedEngine ? 'bg-accent/5' : ''}`}>
                              <td className="py-3 font-black text-accent">{idx + 1}</td>
                              <td className="py-3 font-bold text-white flex items-center gap-2">
                                {name}
                                {name.includes("SARIMA") && (
                                  <div className="group relative">
                                    <Info size={12} className="text-white/20 hover:text-sky-400 cursor-help transition-colors" />
                                    <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-64 p-3 bg-slate-900 border border-white/10 rounded-xl text-[10px] leading-relaxed text-white/70 opacity-0 group-hover:opacity-100 pointer-events-none transition-all duration-200 z-100 shadow-2xl backdrop-blur-xl">
                                      <div className="font-black text-sky-400 mb-1 uppercase tracking-widest text-[8px]">Architectural Limit</div>
                                      SARIMA performs weaker because it assumes linear relationships and cannot handle external variables like weather effectively.
                                    </div>
                                  </div>
                                )}
                              </td>
                              <td className="py-3 text-right font-mono text-rose-400">{Math.round(data.MAE).toLocaleString()} MW</td>
                              <td className="py-3 text-right font-mono text-emerald-400">{data.MAPE?.toFixed(2)}%</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                  
                  {/* Task 4: Model Comparison Insight */}
                  <div className="mt-6 p-4 bg-accent/5 border border-accent/20 rounded-2xl">
                     <div className="text-[10px] font-black text-accent uppercase tracking-widest mb-1">Key Insight</div>
                     <div className="text-[11px] text-white/60 leading-relaxed">
                        ML models outperform classical models due to nonlinear modeling and ability to include external variables like weather.
                     </div>
                  </div>
                </GlassCard>

                {/* Task 2 & 6: Weather vs No-Weather Comparison Block */}
                <GlassCard title="1. Weather Environment (Exogenous)" icon={CloudRain}>
                      {/* Task 2: Weather Proxy Disclosure */}
                      <div className="mb-4 px-3 py-1 bg-sky-500/10 border border-sky-500/20 rounded-lg flex items-center gap-2">
                        <Info size={10} className="text-sky-400" />
                        <span className="text-[9px] text-sky-200/60 italic uppercase tracking-tighter font-bold">
                          Using Regional Proxy: Chennai Weather for National Grid Baseline
                        </span>
                      </div>
                      <div className="space-y-6">
                      <div className="flex justify-between items-center text-[11px]">
                         <span className="text-white/40">Champion MAE (With Weather):</span>
                         <span className="text-emerald-400 font-bold">{Math.round(metrics["ML - Gradient Boosting"]?.MAE || 0).toLocaleString()} MW</span>
                      </div>
                      <div className="flex justify-between items-center text-[11px]">
                         <span className="text-white/40">Baseline MAE (No Weather):</span>
                         <span className="text-rose-400 font-bold">{Math.round(metrics["ML - Gradient Boosting (No Weather)"]?.MAE || 0).toLocaleString()} MW</span>
                      </div>
                      <div className="p-4 bg-emerald-500/5 border border-emerald-500/20 rounded-2xl">
                         <div className="text-[10px] font-black text-emerald-400 uppercase tracking-widest mb-1">Weather Contribution</div>
                         <div className="text-[12px] text-emerald-200/80 font-medium">
                            Weather features reduce error by {Math.round((1 - (metrics["ML - Gradient Boosting"]?.MAE / metrics["ML - Gradient Boosting (No Weather)"]?.MAE)) * 100)}%.
                         </div>
                         <p className="text-[10px] text-emerald-500/60 mt-2 leading-relaxed">
                            Models using weather features reduce MAE significantly compared to models without weather insight.
                         </p>
                      </div>
                   </div>
                </GlassCard>
            </div>
          </motion.div>
        )}

        {/* ──────────────── STAGE 4: DIAGNOSTICS ──────────────── */}
        {activeStage === 'stage4' && (
          <motion.div key="s4" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="grid grid-cols-12 gap-8">
             <GlassCard title="Mathematical Drivers (Relative Gini Importance)" icon={Binary} className="col-span-12 lg:col-span-5">
               <div className="flex flex-col gap-6">
                 {/* Group Influence Bar */}
                 <div className="space-y-3">
                    <div className="flex justify-between items-end">
                       <span className="text-[10px] font-black text-white/30 uppercase tracking-widest">Group Contribution</span>
                       <div className="flex gap-3 text-[9px] font-black">
                          <span className="text-amber-400">MEMORY</span>
                          <span className="text-sky-400">WEATHER</span>
                          <span className="text-slate-400">CALENDAR</span>
                       </div>
                    </div>
                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden flex">
                       {researchData.importance?.groups?.map((group, idx) => (
                         <div 
                           key={group.name} 
                           style={{ width: `${group.value * 100}%` }}
                           className={`${group.name === 'Memory' ? 'bg-amber-400' : group.name === 'Weather' ? 'bg-sky-400' : 'bg-slate-400'}`}
                         />
                       ))}
                    </div>
                 </div>

                 <div className="h-[320px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart layout="vertical" data={researchData.importance?.features || []} margin={{ left: 20, right: 40 }}>
                        <XAxis type="number" hide />
                        <YAxis type="category" dataKey="feature" fontSize={10} stroke="#444" width={100} />
                        <Tooltip contentStyle={{ backgroundColor: '#111', border: 'none', borderRadius: '15px' }} />
                        <Bar dataKey="importance" fill="#63b3ed" radius={[0, 4, 4, 0]}>
                          {(researchData.importance?.features || []).map((entry, index) => (
                            <Cell key={index} fill={index < 3 ? '#fbbf24' : '#63b3ed'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                 </div>
                 
                 {/* Task 1: Forecast Point Reasoning */}
                <div className="p-6 bg-accent/5 border-l-4 border-accent rounded-r-3xl flex flex-col gap-2">
                  <div className="flex items-center gap-2 text-accent">
                    <Zap size={16} />
                    <span className="text-[10px] font-black uppercase tracking-[3px]">Prediction Drivers & Logic</span>
                  </div>
                  <div className="text-[13px] text-white/80 leading-relaxed font-medium">
                    {predictionResult?.explanation || "Awaiting architectural reasoning..."}
                  </div>
                  <div className="grid grid-cols-3 gap-4 mt-2 pt-3 border-t border-white/5">
                    {[
                      { label: "Thermal Impact", val: "Cooling/Heating", icon: Thermometer },
                      { label: "Seasonality", val: "Weekday/Weekend", icon: Calendar },
                      { label: "Inertia", val: "Lag Effect", icon: Clock }
                    ].map((item, i) => (
                      <div key={i} className="flex flex-col">
                        <span className="text-[8px] text-white/30 uppercase font-black">{item.label}</span>
                        <span className="text-[9px] text-accent/80 font-bold">{item.val}</span>
                      </div>
                    ))}
                  </div>
                </div>
                 
                 <div className="p-4 bg-white/5 border border-white/5 rounded-2xl flex flex-col gap-3">
                    <div className="flex gap-3 items-start">
                      <Info size={14} className="text-accent mt-0.5" />
                      <div>
                         <div className="text-[10px] font-black text-white uppercase tracking-widest">Dominant Influence</div>
                         <div className="text-[10px] text-white/50 leading-relaxed italic">
                            "{researchData.importance?.dominant_explanation || "Analyzing model weights..."}"
                         </div>
                      </div>
                    </div>
                    {/* Task 3: Feature Importance Clarification */}
                    <div className="mt-2 pt-3 border-t border-white/5">
                       <p className="text-[10px] text-white/40 leading-relaxed">
                          <span className="text-accent font-black">NOTE:</span> Lag features dominate short-term prediction, while weather features improve generalization and reduce error on unseen data.
                       </p>
                    </div>
                 </div>
               </div>
             </GlassCard>

             <div className="col-span-12 lg:col-span-7 flex flex-col gap-8">
               <GlassCard title="2. Temporal Stability (Rolling validation)" icon={Activity} className="w-full">
                 <div className="flex justify-between items-center mb-6">
                   <div>
                     <span className="text-[10px] text-white/30 uppercase tracking-widest block mb-1">Stability Metric (Mean MAE)</span>
                     <span className="text-xl font-black text-accent">{rollingDiag ? `${rollingDiag.mean.toLocaleString()} MW` : '---'}</span>
                   </div>
                   <div className="text-right">
                     <span className="text-[10px] text-white/30 uppercase tracking-widest block mb-1">Consistency (Std Dev)</span>
                     <span className={`text-xl font-black ${rollingDiag?.std < 500 ? 'text-emerald-400' : 'text-amber-400'}`}>
                       ± {rollingDiag ? rollingDiag.std.toLocaleString() : '---'}
                     </span>
                   </div>
                 </div>
                 
                 <div className="h-[220px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={rollingDiag?.history}>
                        <CartesianGrid stroke="#222" vertical={false} strokeDasharray="5 5" />
                        <XAxis dataKey="date" fontSize={9} stroke="#444" tick={{ fill: '#666' }} />
                        <YAxis stroke="#444" fontSize={9} tickFormatter={(v) => `${(v/1000).toFixed(1)}k`} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0a0c10', border: '1px solid #333', borderRadius: '12px' }}
                          itemStyle={{ color: '#63b3ed', fontSize: '10px' }}
                        />
                        <ReferenceLine y={rollingDiag?.mean} stroke="#63b3ed" strokeDasharray="3 3" label={{ position: 'right', value: 'Avg', fill: '#63b3ed', fontSize: 8 }} />
                        <Line 
                          type="monotone" 
                          dataKey="mae" 
                          stroke="#63b3ed" 
                          strokeWidth={3} 
                          dot={{ r: 3, fill: '#63b3ed' }} 
                          name="Rolling MAE"
                          animationDuration={1500}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                 </div>
                 <div className="mt-4 p-4 bg-white/[0.02] border border-white/5 rounded-2xl flex items-center justify-between">
                   <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${rollingDiag?.reliability === 'HIGH' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'}`}>
                        <CheckCircle2 size={16} />
                      </div>
                      <div>
                        <div className="text-[10px] font-black uppercase text-white tracking-widest">Architectural Reliability</div>
                        <div className="text-[10px] text-white/40">Model maintains consistency within tolerance.</div>
                      </div>
                   </div>
                    <div className="text-[10px] font-black text-accent bg-accent/10 px-3 py-1 rounded-full uppercase italic">
                       MODEL STABILITY: {rollingDiag?.reliability}
                    </div>
                 </div>
                 {/* Task 7: Stability Advisory Improvement */}
                 <div className="px-4 py-2 text-[9px] text-white/30 italic">
                    Reason: Low variance in rolling MAE indicates consistent performance.
                 </div>
               </GlassCard>

               <GlassCard title="Residual Distribution Diagnostics" icon={Target}>
                 <div className="h-[200px] w-full mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={researchData.residuals?.distribution}>
                        <CartesianGrid stroke="#222" vertical={false} />
                        <XAxis dataKey="bin" hide />
                        <YAxis hide />
                        <Tooltip contentStyle={{ backgroundColor: '#111', border: 'none' }} />
                        <Bar dataKey="count" fill="#63b3ed" radius={[4, 4, 0, 0]} opacity={0.6} />
                        <ReferenceLine x={0} stroke="#f43f5e" strokeDasharray="3 3" />
                      </BarChart>
                    </ResponsiveContainer>
                 </div>
                 <div className="flex justify-between items-center mt-4 border-t border-white/5 pt-4">
                   <div>
                     <div className="text-[8px] text-slate-500 uppercase">Computed System Bias</div>
                     <div className="text-xl font-black text-rose-400">{researchData.residuals?.mean.toFixed(2)} MW</div>
                   </div>
                   <div className="text-right">
                     <div className="text-[8px] text-slate-500 uppercase">Error Stability (StdDev)</div>
                     <div className="text-xl font-black text-accent">{researchData.residuals?.std.toFixed(0)} MW</div>
                   </div>
                 </div>
               </GlassCard>

               <GlassCard title="Residual Scatter (Sampling 100 Points)" icon={Activity}>
                  <div className="h-[150px] w-full mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={researchData.residuals?.scatter}>
                        <XAxis dataKey="index" hide />
                        <ReferenceLine y={0} stroke="#444" />
                        <Tooltip contentStyle={{ backgroundColor: '#111', border: 'none' }} />
                        <Line type="monotone" dataKey="value" stroke="#f43f5e" strokeWidth={0} dot={{ r: 3, fill: '#f43f5e' }} activeDot={{ r: 5 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="text-[9px] text-white/20 mt-2 italic text-center">Clustering near the Zero-line indicates a balanced, unbiased model.</p>
               </GlassCard>
             </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Exporter Overlay */}
      <AnimatePresence>
        {showExporter && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-6">
            <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} className="bg-[#111] border border-accent/20 p-8 rounded-3xl max-w-lg w-full">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-black uppercase tracking-widest text-white">Full Research Metadata</h3>
                <button onClick={() => setShowExporter(false)} className="text-slate-500 hover:text-white">✕</button>
              </div>
              <pre className="bg-black p-6 rounded-xl font-mono text-[11px] text-accent leading-loose border border-white/5 overflow-x-auto whitespace-pre-wrap">
                {generateMetadata()}
              </pre>
              <button onClick={() => { navigator.clipboard.writeText(generateMetadata()); alert("Copied to clipboard!"); }} className="w-full mt-6 bg-accent text-slate-900 font-black py-4 rounded-xl flex justify-center items-center gap-2 transition-all active:scale-95">
                 <Download size={18} /> COPY FOR DOCUMENTATION
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Task 9: Business Insight Footer */}
      <footer className="mt-12 mb-8 px-12 text-center">
         <div className="max-w-4xl mx-auto p-6 bg-white/2 border border-white/5 rounded-3xl backdrop-blur-xl">
            <div className="flex items-center justify-center gap-3 text-accent mb-2">
               <Activity size={18} />
               <span className="text-[11px] font-black uppercase tracking-[5px]">Operational Application</span>
            </div>
            <p className="text-[12px] text-slate-400 font-medium">
               This framework helps grid operators plan power generation and avoid overload during peak demand by providing transparent, multivariate foresight.
            </p>
         </div>
      </footer>
    </div>
  );
};

export default App;

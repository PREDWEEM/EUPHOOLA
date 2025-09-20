# -*- coding: utf-8 -*-
# streamlit_emergencia_app_persistente_csv_only.py
# EUPHO – Serie persistente (solo lee meteo_history.csv), SIN reindex (no inventa fechas),
# ventana de gráficos fija: 2025-09-01 → 2026-01-01
# EMERREL: eje Y fijo 0–0.08 · EMEAC: eje Y fijo 0–100%

from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ================== Config de página ==================
st.set_page_config(page_title="Predicción de Emergencia Agrícola EUPHO – OLAVARRIA", layout="wide")

# ================== Constantes visuales y modelo ==================
THR_BAJO_MEDIO = 0.02
THR_MEDIO_ALTO = 0.079
COLOR_MAP = {"Bajo": "#2ca02c", "Medio": "#ff7f0e", "Alto": "#d62728"}
COLOR_FALLBACK = "#808080"

EMEAC_MIN_DEN = 5.0   # Banda inferior
EMEAC_MAX_DEN = 15.0  # Banda superior

# Ventana conceptual (para recorte y para los ejes de los gráficos)
VENTANA_MIN = pd.Timestamp("2025-09-01")
VENTANA_MAX = pd.Timestamp("2026-01-01")
START_SERIE = VENTANA_MIN
END_SERIE   = VENTANA_MAX
st.caption(f"Ventana fija en gráficos: {START_SERIE.date()} → {END_SERIE.date()} (se dibujan solo fechas con datos reales)")

# Archivo local para lectura (única fuente)
HISTORY_PATH = Path("meteo_history.csv")

# ================== Modelo ANN ==================
class PracticalANNModel:
    def __init__(self):
        # Pesos/normalización (ajustar si corresponde a tu último entrenamiento)
        self.IW = np.array([
            [-2.924160, -7.896739, -0.977000, 0.554961, 9.510761, 8.739410, 10.592497, 21.705275, -2.532038, 7.847811,
             -3.907758, 13.933289, 3.727601, 3.751941, 0.639185, -0.758034, 1.556183, 10.458917, -1.343551, -14.721089],
            [0.115434, 0.615363, -0.241457, 5.478775, -26.598709, -2.316081, 0.545053, -2.924576, -14.629911, -8.916969,
             3.516110, -6.315180, -0.005914, 10.801424, 4.928928, 1.158809, 4.394316, -23.519282, 2.694073, 3.387557],
            [6.210673, -0.666815, 2.923249, -8.329875, 7.029798, 1.202168, -4.650263, 2.243358, 22.006945, 5.118664,
             1.901176, -6.076520, 0.239450, -6.862627, -7.592373, 1.422826, -2.575074, 5.302610, -6.379549, -14.810670],
            [10.220671, 2.665316, 4.119266, 5.812964, -3.848171, 1.472373, -4.829068, -7.422444, 0.862384, 0.001028,
             0.853059, 2.953289, 1.403689, -3.040909, -6.946802, -1.799923, 0.994357, -5.551789, -0.764891, 5.520776]
        ], dtype=float)
        self.bias_IW = np.array([
            7.229977, -2.428431, 2.973525, 1.956296, -1.155897, 0.907013, 0.231416, 5.258464, 3.284862, 5.474901,
            2.971978, 4.302273, 1.650572, -1.768043, -7.693806, -0.010850, 1.497102, -2.799158, -2.366918, -9.754413
        ], dtype=float)
        self.LW = np.array([
            5.508609, -21.909052, -10.648533, -2.939799, 8.192068, -2.157424, -3.373238, -5.932938, -2.680237,
            -3.399422, 5.870659, -1.720078, 7.134293, 3.227154, -5.039080, -10.872101, -6.569051, -8.455429,
            2.703778, 4.776029
        ], dtype=float)
        self.bias_out = -5.394722

        # Orden esperado y normalización (Julian_days, TMAX, TMIN, Prec)
        self.input_min = np.array([1.0, 7.7, -3.5, 0.0], dtype=float)
        self.input_max = np.array([148.0, 38.5, 23.5, 59.9], dtype=float)

    def tansig(self, x): return np.tanh(x)

    def normalize_input(self, X_real):
        Xc = np.clip(X_real, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / (self.input_max - self.input_min) - 1

    def desnormalize_output(self, y_norm, ymin=-1.0, ymax=1.0):
        return (y_norm - ymin) / (ymax - ymin)

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)

    def predict(self, X_real):
        X_norm = self.normalize_input(X_real.astype(float))
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm], dtype=float)

        # Desnormalización y protección de dominios
        emerrel_desnorm = np.clip(self.desnormalize_output(emerrel_pred), 0.0, 1.0)

        # Acumulado (0–1) protegido
        emerrel_cumsum = np.clip(np.cumsum(emerrel_desnorm), 0.0, None)

        # Normalización para EMEAC (ajustar a tu validación)
        valor_max_emeac = 8.05
        emer_ac = np.clip(emerrel_cumsum / max(valor_max_emeac, 1e-9), 0.0, 1.0)

        # Incremento diario de EMEAC (0–1)
        emerrel_diff = np.diff(emer_ac, prepend=0.0)
        emerrel_diff = np.clip(emerrel_diff, 0.0, 1.0)

        def clasificar(v):
            if v < THR_BAJO_MEDIO: return "Bajo"
            elif v <= THR_MEDIO_ALTO: return "Medio"
            else: return "Alto"

        riesgo = np.array([clasificar(v) for v in emerrel_diff], dtype=object)
        return pd.DataFrame({"EMERREL(0-1)": emerrel_diff, "Nivel_Emergencia_relativa": riesgo})

@st.cache_resource
def get_model():
    return PracticalANNModel()

modelo = get_model()

# ================== Helpers CSV ==================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Acepta columnas en may/min: Fecha/fecha, TMAX/tmax, TMIN/tmin, Prec/prec, Julian_days/julian_days"""
    rename_map = {}
    cols = {c.lower(): c for c in df.columns}
    if "fecha" in cols:       rename_map[cols["fecha"]] = "Fecha"
    if "tmax" in cols:        rename_map[cols["tmax"]] = "TMAX"
    if "tmin" in cols:        rename_map[cols["tmin"]] = "TMIN"
    if "prec" in cols:        rename_map[cols["prec"]] = "Prec"
    if "julian_days" in cols: rename_map[cols["julian_days"]] = "Julian_days"
    return df.rename(columns=rename_map)

def load_history_strict() -> pd.DataFrame:
    """Lee exclusivamente meteo_history.csv. No reindexa, no inventa fechas."""
    if not HISTORY_PATH.exists():
        st.error(f"No se encontró {HISTORY_PATH.name}. Colocá el archivo en el directorio de la app.")
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

    try:
        # Intento 1: respetando tipos
        dfh = pd.read_csv(HISTORY_PATH, parse_dates=["Fecha"])
    except Exception:
        # Intento 2: leer crudo y normalizar nombres luego
        dfh = pd.read_csv(HISTORY_PATH)
        dfh = _normalize_cols(dfh)
        if "Fecha" in dfh.columns:
            dfh["Fecha"] = pd.to_datetime(dfh["Fecha"], errors="coerce")

    dfh = _normalize_cols(dfh)
    # Sanitizado suave
    if "Fecha" in dfh.columns:
        dfh["Fecha"] = pd.to_datetime(dfh["Fecha"], errors="coerce").dt.normalize()
    for c in ["TMAX","TMIN","Prec","Julian_days"]:
        if c in dfh.columns:
            dfh[c] = pd.to_numeric(dfh[c], errors="coerce")
    if "Prec" in dfh.columns:
        dfh["Prec"] = dfh["Prec"].fillna(0).clip(lower=0)

    # Si no hay Julian_days, lo calculamos desde base
    base = pd.Timestamp("2025-09-01")
    if "Julian_days" not in dfh.columns or dfh["Julian_days"].isna().all():
        if "Fecha" in dfh.columns:
            dfh["Julian_days"] = (dfh["Fecha"] - base).dt.days + 1

    # Ordenar + deduplicar por fecha (sin crear nuevas fechas)
    keep_cols = [c for c in ["Fecha","Julian_days","TMAX","TMIN","Prec"] if c in dfh.columns]
    dfh = (dfh[keep_cols]
           .dropna(subset=["Fecha"])
           .drop_duplicates("Fecha")
           .sort_values("Fecha")
           .reset_index(drop=True))

    return dfh

# ================== App (UI) ==================
st.title("Predicción de Emergencia Agrícola EUPHO – OLAVARRIA")

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral ajustable de EMEAC para 100%", 5.0, 15.0, 14.0, 0.01, format="%.2f"
)
with st.sidebar:
    if st.button("🔄 Forzar refresco/limpieza de caché"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()

def procesar_y_mostrar(df: pd.DataFrame, nombre: str):
    req = {"Julian_days","TMAX","TMIN","Prec"}
    if not req.issubset(df.columns):
        st.warning(f"{nombre}: faltan columnas {req - set(df.columns)}")
        return

    # Asegurar Fecha desde julianos si no vino explícita
    if "Fecha" not in df.columns:
        base = pd.Timestamp("2025-09-01")
        jd = pd.to_numeric(df["Julian_days"], errors="coerce")
        df["Fecha"] = (base + pd.to_timedelta(jd - 1, unit="D")).dt.normalize()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    # Ordenar + deduplicar por fecha (sin crear nuevas fechas)
    df = (
        df.sort_values("Fecha")
          .drop_duplicates("Fecha")
          .reset_index(drop=True)
    )

    # Recorte por ventana conceptual (sin reindex; NO inventar días)
    m_vis = (df["Fecha"] >= START_SERIE) & (df["Fecha"] <= END_SERIE)
    df_vis = df.loc[m_vis].copy()

    if df_vis.empty:
        st.warning(
            f"{nombre}: no hay datos dentro de la ventana "
            f"{START_SERIE.date()} → {END_SERIE.date()}. "
            "Revisá que el CSV tenga fechas en ese rango."
        )
        return

    # Sanitizado suave (sin inventar valores)
    for c in ["TMAX","TMIN","Prec"]:
        df_vis[c] = pd.to_numeric(df_vis[c], errors="coerce")
    df_vis["Prec"] = df_vis["Prec"].fillna(0).clip(lower=0)

    # Recalcular julianos por consistencia
    base = pd.Timestamp("2025-09-01")
    df_vis["Julian_days"] = (df_vis["Fecha"] - base).dt.days + 1

    # ====== Modelo ======
    X_real = df_vis[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)
    fechas = df_vis["Fecha"]
    pred = modelo.predict(X_real)

    pred["Fecha"] = fechas
    pred["Julian_days"] = df_vis["Julian_days"].to_numpy()
    pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()

    # Bandas EMEAC
    pred["EMEAC (0-1) - mínimo"]    = pred["EMERREL acumulado"] / EMEAC_MIN_DEN
    pred["EMEAC (0-1) - máximo"]    = pred["EMERREL acumulado"] / EMEAC_MAX_DEN
    pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
    for col in ["EMEAC (0-1) - mínimo","EMEAC (0-1) - máximo","EMEAC (0-1) - ajustable"]:
        pred[col.replace("(0-1)","(%)")] = (pred[col]*100).clip(0,100)

    # Regla estética: cuando EMEAC ajustable < 10%, forzar "Bajo"
    pred.loc[pred["EMEAC (%) - ajustable"] < 10.0, "Nivel_Emergencia_relativa"] = "Bajo"

    pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(5, 1).mean()

    # ====== Figuras (ejes X fijos a la ventana completa) ======
    st.subheader(f"EMERGENCIA RELATIVA DIARIA — {nombre}")
    colores = pred["Nivel_Emergencia_relativa"].map(COLOR_MAP).fillna(COLOR_FALLBACK).tolist()
    fig_er = go.Figure()
    fig_er.add_bar(
        x=pred["Fecha"],
        y=pred["EMERREL(0-1)"],
        marker=dict(color=colores),
        customdata=pred["Nivel_Emergencia_relativa"],
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
        name="EMERREL"
    )
    fig_er.add_scatter(x=pred["Fecha"], y=pred["EMERREL_MA5"], mode="lines", name="MA5")
    fig_er.update_xaxes(
        range=[str(VENTANA_MIN.date()), str(VENTANA_MAX.date())],
        dtick="M1", tickformat="%b"
    )
    fig_er.update_yaxes(range=[0, 0.08])  # EMERREL: Y fijo 0–0.08
    st.plotly_chart(fig_er, use_container_width=True)

    st.subheader(f"EMERGENCIA ACUMULADA DIARIA — {nombre}")
    fig_acc = go.Figure()
    fig_acc.add_scatter(x=pred["Fecha"], y=pred["EMEAC (%) - mínimo"], mode="lines", line=dict(width=0), name="EMEAC mín")
    fig_acc.add_scatter(x=pred["Fecha"], y=pred["EMEAC (%) - máximo"], mode="lines", line=dict(width=0), fill="tonexty", name="EMEAC máx")
    fig_acc.add_scatter(x=pred["Fecha"], y=pred["EMEAC (%) - ajustable"], mode="lines", line=dict(width=2.5), name=f"Ajustable /{umbral_usuario:.2f}")
    fig_acc.update_traces(hovertemplate="%{x|%d-%b-%Y}<br>%{y:.1f}%<extra></extra>")
    fig_acc.update_yaxes(range=[0, 100])  # EMEAC: Y fijo 0–100%
    fig_acc.update_xaxes(
        range=[str(VENTANA_MIN.date()), str(VENTANA_MAX.date())],
        dtick="M1", tickformat="%b"
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # ====== Tabla ======
    st.subheader(f"Resultados (solo fechas con datos) — {nombre}")
    tabla = pred[["Fecha","Julian_days","Nivel_Emergencia_relativa"]].copy()
    tabla["EMEAC (%)"] = pred["EMEAC (%) - ajustable"]
    iconos = {"Bajo": "🟢 Bajo", "Medio": "🟠 Medio", "Alto": "🔴 Alto"}
    tabla["Nivel_Emergencia_relativa"] = tabla["Nivel_Emergencia_relativa"].map(iconos)
    tabla = tabla.rename(columns={"Nivel_Emergencia_relativa": "Nivel de EMERREL"})
    st.dataframe(tabla, use_container_width=True)

    st.download_button(
        "Descargar CSV",
        tabla.to_csv(index=False).encode("utf-8"),
        f"{nombre}_resultados.csv",
        "text/csv"
    )

# ================== Flujo principal ==================
st.markdown("—")

# 1) Leer EXCLUSIVAMENTE meteo_history.csv
df_hist = load_history_strict()

# 2) Mostrar usando la serie consolidada tal cual (no se borran fechas antiguas; no se crean fechas nuevas)
if df_hist.empty:
    st.stop()
else:
    st.success(f"History: {df_hist['Fecha'].min().date()} → {df_hist['Fecha'].max().date()} · {len(df_hist)} día(s)")
    procesar_y_mostrar(df_hist, "meteo_history.csv")

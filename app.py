from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="CardWise – Customer Segmentation", page_icon="💳", layout="wide")

# RTL לטקסט בעברית, קוד וטבלאות נשארים LTR
st.markdown(
    """
    <style>
    .stMarkdown, .stAlert, h1, h2, h3, h4, p, li, label { direction: rtl; text-align: right; }
    code, pre, .stCode, .stDataFrame, table { direction: ltr !important; text-align: left !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).parent / "Customer Data.csv"
RANDOM_STATE = 42

MONEY_COLS = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
]
FREQ_COLS = [
    "BALANCE_FREQUENCY", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT",
]
FEATURES = MONEY_COLS + FREQ_COLS
DROPPED = {
    "CUST_ID": "מזהה בלבד – אין לו משמעות התנהגותית.",
    "TENURE": "כ־85% מהלקוחות עם ערך 12 – כמעט ללא שונות, ולכן לא תורם להפרדה.",
    "PURCHASES_TRX": "מתואם מאוד עם PURCHASES_FREQUENCY ועם PURCHASES – מידע כפול.",
    "CASH_ADVANCE_TRX": "מתואם מאוד עם CASH_ADVANCE_FREQUENCY – מידע כפול.",
}
KEY_COLS = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]


@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def clean(df):
    df = df.copy()
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())
    return df


@st.cache_data
def scale(df):
    X = pd.concat([np.log1p(df[MONEY_COLS]), df[FREQ_COLS]], axis=1)
    return StandardScaler().fit_transform(X)


@st.cache_data
def elbow(X, k_max=10):
    rows = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=4000, random_state=RANDOM_STATE)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


@st.cache_data
def fit_kmeans(X, k):
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)


def name_clusters(profile):
    """נותן שם עסקי לכל קלאסטר לפי המאפיינים הבולטים שלו (עבור k=4)."""
    names = {}
    remaining = list(profile.index)
    rules = [
        ("Cash-Advance Seekers", lambda p: p["CASH_ADVANCE_FREQUENCY"].idxmax()),
        ("Dormant / Low-Engagement", lambda p: p["BALANCE_FREQUENCY"].idxmin()),
        ("High-Value Frequent Spenders", lambda p: p["ONEOFF_PURCHASES"].idxmax()),
        ("Installment Shoppers", lambda p: p["PURCHASES_INSTALLMENTS_FREQUENCY"].idxmax()),
    ]
    for label, pick in rules:
        c = pick(profile.loc[remaining])
        names[c] = label
        remaining.remove(c)
    return names


def show_fig(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


raw = load_raw()
df = clean(raw)
X = scale(df)

# ---------------------------------------------------------------- כותרת
st.title("💳 CardWise – פילוח לקוחות עם K-Means")
st.markdown(
    "ניתוח של בסיס הלקוחות של CardWise (קובץ `Customer Data.csv`) כדי למצוא קבוצות של לקוחות דומים "
    "(segments) בעזרת **KMeans**, לבנות פרופיל לכל קבוצה ולהמליץ לצוות השיווק על מסרים ומבצעים."
)

tabs = st.tabs([
    "א׳ – טעינה ובדיקה", "ב׳ – EDA", "ג׳ – קדם־עיבוד", "ד׳ – Elbow",
    "ה׳ – KMeans ופרופילים", "ו׳ – מסקנות עסקיות",
])

# ---------------------------------------------------------------- חלק א
with tabs[0]:
    st.header("חלק א׳ – טעינת הנתונים ובדיקה ראשונית")

    st.subheader("1. טעינת הקובץ ל־DataFrame")
    st.code('df = pd.read_csv("Customer Data.csv")', language="python")

    st.subheader("2. סקירה ראשונית של הנתונים")

    st.markdown('#### חמש השורות הראשונות <span dir="ltr">(df.head())</span>', unsafe_allow_html=True)
    st.dataframe(raw.head(), width="stretch", hide_index=True)

    st.markdown('#### סוגי עמודות וערכים חסרים <span dir="ltr">(df.info())</span>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("שורות", f"{raw.shape[0]:,}")
    m2.metric("עמודות", raw.shape[1])
    m3.metric("עמודות מספריות", int(raw.select_dtypes("number").shape[1]))
    m4.metric("זיכרון", f"{raw.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
    info_df = pd.DataFrame({
        "Column": raw.columns,
        "Dtype": raw.dtypes.astype(str).values,
        "Non-Null": raw.notna().sum().values,
        "Missing": raw.isna().sum().values,
        "Missing %": (raw.isna().mean() * 100).round(2).values,
        "Unique": raw.nunique().values,
    })
    st.dataframe(
        info_df, width="stretch", hide_index=True,
        column_config={
            "Non-Null": st.column_config.ProgressColumn("Non-Null", format="%d", min_value=0, max_value=len(raw)),
            "Missing %": st.column_config.NumberColumn("Missing %", format="%.2f%%"),
        },
    )

    st.markdown('#### סטטיסטיקה תיאורית <span dir="ltr">(df.describe())</span>', unsafe_allow_html=True)
    desc = raw.describe().T
    desc["count"] = desc["count"].astype(int)
    st.dataframe(
        desc.style.format("{:,.2f}", subset=desc.columns.drop("count"))
        .format("{:,}", subset=["count"])
        .background_gradient(cmap="Blues", subset=["mean", "50%"], axis=0),
        width="stretch",
    )
    st.markdown(
        "- כל העמודות מספריות חוץ מ־`CUST_ID` (טקסט).\n"
        "- בעמודות הכספיות הממוצע גבוה בהרבה מהחציון. לדוגמה, ב־`CASH_ADVANCE` החציון הוא 0 והממוצע כ־979. "
        "זה סימן לא־סימטריה חזקה (זנב ימני) ולערכים קיצוניים.\n"
        "- `TENURE` כמעט קבוע: רוב הלקוחות מחזיקים בכרטיס 12 חודשים."
    )

    st.subheader("3. ערכים חסרים וטיפול בהם")
    na = raw.isna().sum()
    na = na[na > 0].to_frame("missing")
    na["% מהשורות"] = (na["missing"] / len(raw) * 100).round(2)
    st.dataframe(na, width="stretch")
    st.markdown(
        "**שיטת הטיפול: השמה בערך החציון (median imputation).**\n\n"
        "- `MINIMUM_PAYMENTS`: חסרים 313 ערכים (כ־3.5%). מחיקה הייתה מוציאה מהניתוח לקוחות אמיתיים, "
        "ולכן עדיף להשלים את הערכים.\n"
        "- `CREDIT_LIMIT`: חסר ערך אחד בלבד, ולכן משלימים אותו באותה שיטה.\n"
        "- בחרנו **בחציון ולא בממוצע** כי ההתפלגויות מוטות מאוד ימינה. "
        "הממוצע מושפע מאאוטליירים והיה מנפח את הערכים שמשלימים, והחציון עמיד להם.\n"
        "- אין כפילויות ב־`CUST_ID`."
    )
    st.success(f"אחרי ההשלמה: {int(df.isna().sum().sum())} ערכים חסרים.")

# ---------------------------------------------------------------- חלק ב
with tabs[1]:
    st.header("חלק ב׳ – ניתוח חקר נתונים (EDA)")

    st.subheader("1. מטריצת קורלציה בין התכונות העיקריות")
    corr = df[KEY_COLS].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation – key features")
    show_fig(fig)
    st.markdown(
        f"""
**פרשנות:**
- **PURCHASES ↔ PAYMENTS ({corr.loc['PURCHASES', 'PAYMENTS']:.2f})**: הקשר החזק ביותר. מי שקונה הרבה גם משלם הרבה, וזה מתאים ל"מוציאים" שמחזירים את החוב.
- **BALANCE ↔ CREDIT_LIMIT ({corr.loc['BALANCE', 'CREDIT_LIMIT']:.2f})** ו־**BALANCE ↔ CASH_ADVANCE ({corr.loc['BALANCE', 'CASH_ADVANCE']:.2f})**: יתרה גבוהה קשורה למסגרת גבוהה ולמשיכות מזומן. כלומר, חלק מהלקוחות "מגלגלים" חוב שנוצר ממשיכות מזומן.
- **PURCHASES ↔ CASH_ADVANCE ({corr.loc['PURCHASES', 'CASH_ADVANCE']:.2f})**: כמעט אין קשר. **רכישות ומשיכות מזומן הן שני דפוסי שימוש נפרדים**, וזה רמז ראשון לפילוח שנקבל.
- **CREDIT_LIMIT** מתואם בצורה מתונה עם כל השאר, כי מסגרת גבוהה מאפשרת גם הוצאות וגם יתרה גבוהה.
"""
    )
    with st.expander("מטריצת קורלציה מלאה (כל התכונות)"):
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(df.drop(columns="CUST_ID").corr(), cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        show_fig(fig)

    st.subheader("2. היסטוגרמות – BALANCE ו־PURCHASES")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["BALANCE", "PURCHASES"]):
        sns.histplot(df[col], bins=60, kde=True, ax=ax)
        ax.set_title(f"{col}  (skew = {df[col].skew():.2f})")
    show_fig(fig)
    st.markdown(
        f"- שתי ההתפלגויות **מוטות חזק ימינה**: skew של {df['BALANCE'].skew():.1f} ב־BALANCE ושל {df['PURCHASES'].skew():.1f} ב־PURCHASES. "
        "רוב הלקוחות מרוכזים בערכים נמוכים, ויש זנב ארוך של מעט לקוחות עם ערכים גבוהים מאוד (אאוטליירים).\n"
        f"- ל־{(df['PURCHASES'] == 0).mean():.0%} מהלקוחות אין רכישות בכלל (PURCHASES = 0).\n"
        "- **המשמעות למודל:** KMeans מבוסס מרחק אוקלידי ורגיש לערכים קיצוניים. לכן בחלק ג׳ נפעיל טרנספורמציית log על התכונות הכספיות לפני הסטנדרטיזציה."
    )
    with st.expander("אותן התפלגויות אחרי log1p"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col in zip(axes, ["BALANCE", "PURCHASES"]):
            sns.histplot(np.log1p(df[col]), bins=60, kde=True, ax=ax, color="tab:green")
            ax.set_title(f"log1p({col})")
        show_fig(fig)

    st.subheader("Scatter plot – PURCHASES מול CASH_ADVANCE")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["PURCHASES"], df["CASH_ADVANCE"], s=6, alpha=0.4)
    ax.set_xlabel("PURCHASES")
    ax.set_ylabel("CASH_ADVANCE")
    show_fig(fig)
    both = ((df["PURCHASES"] > 0) & (df["CASH_ADVANCE"] > 0)).mean()
    st.markdown(
        "- הנקודות מתרכזות לאורך **שני הצירים**: יש לקוחות שקונים הרבה ולא מושכים מזומן, ויש לקוחות שמושכים הרבה מזומן ולא קונים. "
        f"רק {both:.0%} מהלקוחות משתמשים בשני הדברים.\n"
        "- זה מחזק את מה שראינו בקורלציה: מדובר ב**שני סגמנטים התנהגותיים שונים**."
    )

    st.subheader("3. נקודות קיצון – Boxplot ל־CASH_ADVANCE ול־CREDIT_LIMIT")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    out_rows = []
    for ax, col in zip(axes, ["CASH_ADVANCE", "CREDIT_LIMIT"]):
        sns.boxplot(x=df[col], ax=ax, color="tab:orange")
        ax.set_title(col)
        q1, q3 = df[col].quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        n_out = int((df[col] > upper).sum())
        out_rows.append({"feature": col, "Q1": q1, "Q3": q3, "upper whisker": upper,
                         "outliers": n_out, "% outliers": round(n_out / len(df) * 100, 1)})
    show_fig(fig)
    st.dataframe(pd.DataFrame(out_rows).round(1), width="stretch", hide_index=True)
    st.markdown(
        "- לפי כלל ה־IQR (`Q3 + 1.5·IQR`) יש מאות לקוחות מעל השפם העליון בשתי העמודות.\n"
        "- **החלטה:** לא מוחקים אותם. אלה לקוחות אמיתיים, ולעתים הם בדיוק המעניינים ביותר עסקית "
        "(לקוחות בעלי ערך גבוה, או לקוחות בסיכון). "
        "במקום זה מקטינים את ההשפעה שלהם בעזרת log1p ו־StandardScaler, כך שהמודל לא ייבנה סביבם."
    )

# ---------------------------------------------------------------- חלק ג
with tabs[2]:
    st.header("חלק ג׳ – קדם־עיבוד והתאמה למודל")

    st.subheader("1. בחירת תכונות – מחקר קצר")
    st.markdown(
        """
**מה עולה ממחקר על פילוח לקוחות כרטיסי אשראי:**
- בספרות ובפרויקטים על נתונים כאלה (מבנה מוכר מ־Kaggle בשם *Credit Card Dataset for Clustering*) הפילוח נשען בדרך כלל על שלושה צירים התנהגותיים:
  **(1) היקף הוצאה וסוג רכישה** (חד־פעמית או בתשלומים), **(2) שימוש במזומן ובאשראי מתגלגל** (cash advance ויתרה), ו־**(3) הרגלי החזר** (תשלומים, תשלום מינימום, אחוז תשלום מלא).
- זה מתאים לגישת RFM (Recency, Frequency, Monetary): משתמשים גם בתכונות **כספיות** (Monetary) וגם בתכונות **תדירות** (Frequency).
- המלצות שחוזרות על עצמן: להסיר מזהים, להסיר משתנים כמעט קבועים, לצמצם משתנים כפולים (ספירת עסקאות מול תדירות), ולנרמל משתנים כספיים מוטים בעזרת log לפני KMeans.
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ תכונות שנבחרו (14):**")
        st.markdown("*כספיות (8):* " + ", ".join(f"`{c}`" for c in MONEY_COLS))
        st.markdown("*תדירות והתנהגות תשלום (6):* " + ", ".join(f"`{c}`" for c in FREQ_COLS))
    with c2:
        st.markdown("**❌ תכונות שהוסרו:**")
        for col, why in DROPPED.items():
            st.markdown(f"- `{col}`: {why}")
    trx = df[["PURCHASES_TRX", "PURCHASES_FREQUENCY", "CASH_ADVANCE_TRX", "CASH_ADVANCE_FREQUENCY"]].corr()
    st.caption(
        f"corr(PURCHASES_TRX, PURCHASES_FREQUENCY) = {trx.iloc[0, 1]:.2f} · "
        f"corr(CASH_ADVANCE_TRX, CASH_ADVANCE_FREQUENCY) = {trx.iloc[2, 3]:.2f}"
    )

    st.subheader("2. Scale / Standardization")
    st.markdown(
        "- התכונות בסקאלות שונות לגמרי: תכונות כספיות מגיעות לעשרות אלפים, ותכונות התדירות בין 0 ל־1. "
        "בלי סטנדרטיזציה, KMeans (מרחק אוקלידי) היה מתעלם כמעט לגמרי מתכונות התדירות.\n"
        "- **שלב א:** `log1p` על 8 התכונות הכספיות, כדי לטפל באי־סימטריה ובאאוטליירים מחלק ב׳.\n"
        "- **שלב ב:** `StandardScaler` על כל 14 התכונות, כך שלכל אחת יש ממוצע 0 וסטיית תקן 1."
    )
    st.code(
        "X = pd.concat([np.log1p(df[MONEY_COLS]), df[FREQ_COLS]], axis=1)\n"
        "X_scaled = StandardScaler().fit_transform(X)",
        language="python",
    )
    st.dataframe(
        pd.DataFrame(X, columns=FEATURES).describe().T[["mean", "std", "min", "max"]].round(2),
        width="stretch",
    )

# ---------------------------------------------------------------- חלק ד
with tabs[3]:
    st.header("חלק ד׳ – מציאת מספר הקלאסטרים (Elbow)")
    el = elbow(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(el["k"], el["inertia"], marker="o")
    axes[0].axvline(4, ls="--", color="red")
    axes[0].set(title="Elbow method", xlabel="k", ylabel="Inertia (WCSS)")
    axes[1].plot(el["k"], el["silhouette"], marker="o", color="tab:green")
    axes[1].axvline(4, ls="--", color="red")
    axes[1].set(title="Silhouette score", xlabel="k", ylabel="silhouette")
    show_fig(fig)
    el_show = el.copy()
    el_show["ירידה ב־inertia (%)"] = (-el["inertia"].pct_change() * 100).round(1)
    st.dataframe(el_show.round(3), width="stretch", hide_index=True)

    st.subheader("1. בחירת k והסבר")
    st.markdown(
        "**k = 4 נבחר.**\n\n"
        "- **Elbow:** הירידה ב־inertia חדה עד k=4 (כ־17% ואחר כך כ־11%), ומשם היא מתמתנת ל־5–9% לכל קלאסטר נוסף. "
        "זה ה'מרפק': קלאסטרים נוספים כמעט לא משפרים את הדחיסות.\n"
        "- **Silhouette:** הציון הגבוה ביותר הוא ב־k=4, כלומר זו ההפרדה הטובה ביותר בין הקבוצות.\n"
        "- **שיקול עסקי:** ב־k=4 הקבוצות מאוזנות (בין כ־1,300 לכ־2,900 לקוחות), ולכל אחת יש פרשנות ברורה שאפשר לפעול לפיה. "
        "מספר גדול יותר יוצר תתי־קבוצות קטנות שקשה לבנות להן קמפיין נפרד."
    )

# ---------------------------------------------------------------- חלק ה
with tabs[4]:
    st.header("חלק ה׳ – אימון KMeans ופרופילינג של קלאסטרים")

    st.subheader("1. אימון המודל")
    st.code("kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)\n"
            "df['Cluster'] = kmeans.fit_predict(X_scaled)", language="python")
    df["Cluster"] = fit_kmeans(X, 4)

    st.subheader("2. עמודת Cluster ב־DataFrame")
    st.dataframe(df[["CUST_ID", "Cluster"] + KEY_COLS].head(10), width="stretch", hide_index=True)

    st.subheader("3. מאפיינים ממוצעים לכל קלאסטר (groupby)")
    profile_all = df.groupby("Cluster")[FEATURES].mean()
    names = name_clusters(profile_all)
    df["Segment"] = df["Cluster"].map(names)

    summary = df.groupby("Cluster").agg(
        Segment=("Segment", "first"),
        Size=("CUST_ID", "count"),
        **{c: (c, "mean") for c in KEY_COLS},
    )
    summary.insert(2, "% of customers", (summary["Size"] / len(df) * 100).round(1))
    st.dataframe(summary.round(0).astype({"% of customers": float}), width="stretch")
    st.download_button("⬇️ הורדת הקובץ עם עמודת Cluster (CSV)",
                       df.to_csv(index=False).encode("utf-8"), "customers_with_clusters.csv", "text/csv")

    with st.expander("טבלת ממוצעים מלאה – כל 14 התכונות"):
        st.dataframe(profile_all.T.round(2).rename(columns=names), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        summary.set_index("Segment")["Size"].plot.barh(ax=ax, color=sns.color_palette("Set2", 4))
        ax.set_title("Cluster size")
        show_fig(fig)
    with c2:
        pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=df["Segment"], s=6, alpha=0.5,
                        palette="Set2", ax=ax, linewidth=0)
        ax.set(title="Clusters in PCA space", xlabel="PC1", ylabel="PC2")
        ax.legend(fontsize=7, markerscale=2)
        show_fig(fig)

    z = pd.DataFrame(X, columns=FEATURES).groupby(df["Cluster"]).mean().rename(index=names)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    sns.heatmap(z, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Cluster centroids (standardized) – red = above average, blue = below")
    show_fig(fig)

    st.subheader("4. פרשנות הקלאסטרים")
    p = profile_all.rename(index=names)
    sz = df["Segment"].value_counts()

    def row(seg, col, fmt="{:,.0f}"):
        return fmt.format(p.loc[seg, col])

    seg = "High-Value Frequent Spenders"
    st.markdown(f"#### 💎 {seg} ({sz[seg]:,} לקוחות)")
    st.markdown(
        f"- **הוצאה הגבוהה ביותר**: רכישות ממוצעות של {row(seg, 'PURCHASES')}, משולבות חד־פעמיות ({row(seg, 'ONEOFF_PURCHASES')}) ובתשלומים ({row(seg, 'INSTALLMENTS_PURCHASES')}).\n"
        f"- **תדירות רכישה גבוהה** ({row(seg, 'PURCHASES_FREQUENCY', '{:.2f}')}), **המסגרת הגבוהה ביותר** ({row(seg, 'CREDIT_LIMIT')}) והתשלומים הגבוהים ביותר ({row(seg, 'PAYMENTS')}).\n"
        "- **סגנון:** הלקוחות הרווחיים ביותר. הם משתמשים בכרטיס כאמצעי התשלום העיקרי שלהם, לכל סוגי הקניות."
    )
    seg = "Cash-Advance Seekers"
    st.markdown(f"#### 💵 {seg} ({sz[seg]:,} לקוחות)")
    st.markdown(
        f"- **משיכות מזומן גבוהות מאוד** ({row(seg, 'CASH_ADVANCE')}), פי 2 ויותר מכל קבוצה אחרת, ותדירות משיכה של {row(seg, 'CASH_ADVANCE_FREQUENCY', '{:.2f}')}.\n"
        f"- **כמעט לא קונים בכרטיס** (רכישות ממוצעות {row(seg, 'PURCHASES')}, תדירות {row(seg, 'PURCHASES_FREQUENCY', '{:.2f}')}).\n"
        f"- **היתרה הגבוהה ביותר** ({row(seg, 'BALANCE')}), תשלום מינימום גבוה, ורק {row(seg, 'PRC_FULL_PAYMENT', '{:.0%}')} מהחודשים נסגרים בתשלום מלא.\n"
        "- **סגנון:** משתמשים בכרטיס כמקור אשראי ומזומן. רווחיים מבחינת ריבית, אבל **בסיכון האשראי הגבוה ביותר**."
    )
    seg = "Installment Shoppers"
    st.markdown(f"#### 🛒 {seg} ({sz[seg]:,} לקוחות)")
    st.markdown(
        f"- **קונים כמעט רק בתשלומים**: תדירות תשלומים {row(seg, 'PURCHASES_INSTALLMENTS_FREQUENCY', '{:.2f}')}, לעומת {row(seg, 'ONEOFF_PURCHASES_FREQUENCY', '{:.2f}')} בלבד לעסקאות חד־פעמיות.\n"
        f"- תדירות רכישה גבוהה ({row(seg, 'PURCHASES_FREQUENCY', '{:.2f}')}) אבל בסכומים בינוניים ({row(seg, 'PURCHASES')}), והמסגרת הנמוכה ביותר ({row(seg, 'CREDIT_LIMIT')}).\n"
        f"- כמעט לא מושכים מזומן, ושיעור התשלום המלא הגבוה ביותר ({row(seg, 'PRC_FULL_PAYMENT', '{:.0%}')}).\n"
        "- **סגנון:** צרכנים קבועים ואחראיים שמפזרים קניות לתשלומים. לקוחות יציבים עם פוטנציאל לגדול."
    )
    seg = "Dormant / Low-Engagement"
    st.markdown(f"#### 😴 {seg} ({sz[seg]:,} לקוחות)")
    st.markdown(
        f"- **יתרה זניחה** ({row(seg, 'BALANCE')}), והיתרה מתעדכנת לעתים רחוקות (BALANCE_FREQUENCY של {row(seg, 'BALANCE_FREQUENCY', '{:.2f}')}).\n"
        f"- פעילות נמוכה בכל הצירים: רכישות {row(seg, 'PURCHASES')}, תדירות {row(seg, 'PURCHASES_FREQUENCY', '{:.2f}')}, ומזומן זניח.\n"
        f"- **מסגרת לא מנוצלת**: מסגרת ממוצעת של {row(seg, 'CREDIT_LIMIT')} עם שימוש מינימלי.\n"
        "- **סגנון:** לקוחות \"רדומים\". הכרטיס כנראה משני אצלם או נשכח בארנק. **סיכון נטישה גבוה.**"
    )

# ---------------------------------------------------------------- חלק ו
with tabs[5]:
    st.header("חלק ו׳ – סיכום ומסקנות עסקיות")
    share = (df["Segment"].value_counts(normalize=True) * 100).round(0)

    st.subheader("1. תובנות מהתפלגות הקלאסטרים")
    st.markdown(
        f"""
- **הקבוצה הגדולה ביותר היא Cash-Advance Seekers (כ־{share['Cash-Advance Seekers']:.0f}%)**: כשליש מהלקוחות משתמשים בכרטיס בעיקר למשיכת מזומן ולגלגול יתרה. זה מקור הכנסה מריבית, אבל גם **ריכוז משמעותי של סיכון אשראי** בתיק.
- **רק כ־{share['High-Value Frequent Spenders']:.0f}% הם High-Value Spenders**, והם מייצרים את רוב נפח הרכישות. כדאי להשקיע בשימור שלהם.
- **כ־{share['Installment Shoppers']:.0f}% הם Installment Shoppers**: קבוצה פעילה ואחראית שהמסגרת שלה נמוכה ביחס לפעילות. יש כאן פוטנציאל לגדילה.
- **כ־{share['Dormant / Low-Engagement']:.0f}% רדומים**, עם מסגרת מאושרת שלא מנוצלת. זו עלות הון והזדמנות שמבוזבזת.
- רכישות ומשיכות מזומן הן **שני עולמות כמעט נפרדים**. לכן אסטרטגיה אחת לכל הלקוחות לא תעבוד.
"""
    )

    st.subheader("2. יישום המודל – שיווק ממוקד, ניהול סיכונים ומבצעים")
    reco = pd.DataFrame([
        {"סגמנט": "💎 High-Value Frequent Spenders",
         "מטרה": "שימור והגדלת נאמנות",
         "מסר שיווקי": "\"אתה לקוח VIP – תיהנה יותר מכל עסקה\"",
         "מבצעים / הצעות": "כרטיס פרימיום, cashback מוגדל ונקודות על עסקאות חד־פעמיות, הטבות טיסות ולאונג', הגדלת מסגרת יזומה",
         "ניהול סיכון": "סיכון נמוך: לאשר הגדלות מסגרת במהירות"},
        {"סגמנט": "💵 Cash-Advance Seekers",
         "מטרה": "הפחתת סיכון והמרה למוצר בריא יותר",
         "מסר שיווקי": "\"צריך גמישות? יש לנו דרך זולה יותר\"",
         "מבצעים / הצעות": "הלוואה אישית בריבית נמוכה במקום משיכות מזומן, איחוד חובות, הטבה על מעבר לתשלומים, תמריץ להוראת קבע של תשלום מלא",
         "ניהול סיכון": "ניטור צמוד, הקפאה או הקטנה של מסגרת מזומן כשיש סימני מצוקה, התראות יתרה, ריבית לפי סיכון"},
        {"סגמנט": "🛒 Installment Shoppers",
         "מטרה": "הגדלת הוצאה וחשיפה לרכישות גדולות",
         "מסר שיווקי": "\"הקנייה הגדולה הבאה שלך – ב־12 תשלומים ללא ריבית\"",
         "מבצעים / הצעות": "תשלומים ללא ריבית אצל רשתות שותפות (אלקטרוניקה, ריהוט), הגדלת מסגרת הדרגתית, הטבות על קנייה חוזרת",
         "ניהול סיכון": "סיכון נמוך: מועמדים טובים להגדלת מסגרת (הם משלמים במלואו לעתים קרובות)"},
        {"סגמנט": "😴 Dormant / Low-Engagement",
         "מטרה": "הפעלה מחדש (Re-activation) ומניעת נטישה",
         "מסר שיווקי": "\"התגעגענו! מתנה מחכה לך על העסקה הבאה\"",
         "מבצעים / הצעות": "בונוס על 3 העסקאות הראשונות, פטור מדמי כרטיס, חיבור לארנק דיגיטלי, הטבות על הוצאות יומיומיות (סופר, דלק)",
         "ניהול סיכון": "בחינת הקטנת מסגרות שלא מנוצלות כדי לשחרר הון"},
    ])
    st.dataframe(reco, width="stretch", hide_index=True)

    st.markdown(
        """
**איך מטמיעים את המודל בארגון:**
1. **שיווק ממוקד:** מריצים את המודל מחדש כל חודש על נתוני הלקוחות, ומשייכים כל לקוח לסגמנט במערכת ה־CRM. כל קמפיין מוגדר לפי סגמנט: ערוץ, מסר והצעה שונים, כמו בטבלה.
2. **ניהול סיכוני אשראי:** מעבר של לקוח ל־Cash-Advance Seekers הוא **אות אזהרה מוקדם**. אפשר לחבר אותו למערכת ניהול הסיכונים לבדיקת מסגרת ולפנייה יזומה.
3. **מבצעים מותאמים:** בודקים כל הצעה ב־A/B test בתוך הסגמנט, ומודדים KPI לכל קבוצה: נפח רכישות, שיעור הפעלה מחדש, ירידה במשיכות מזומן, ושיעור תשלום מלא.
4. **מעקב מעברים:** מעקב אחרי לקוחות שעוברים בין סגמנטים לאורך זמן (למשל מ־Dormant ל־Installment) מראה אם הקמפיינים עובדים.
"""
    )

st.divider()
st.caption("CardWise Customer Segmentation · KMeans (k=4, random_state=42) · Streamlit")

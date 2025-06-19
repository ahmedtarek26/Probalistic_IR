# Probabilistic Information Retrieval: BIM and BM25

This project implements and explains two fundamental probabilistic models for Information Retrieval (IR):

* **Binary Independence Model (BIM)**
* **Okapi BM25 Scoring**

We apply these models on the [20 Newsgroups Dataset](https://www.kaggle.com/datasets/crawford/20-newsgroups), and walk through each step with mathematical intuition, code, and example-based explanation.

---

## 📂 Dataset

**20 Newsgroups Dataset**

* \~20,000 newsgroup documents across 20 topics.
* Files are in plain text format.
* Used as a real-world corpus to test and rank documents.

---

## 🚀 Project Objectives

* Implement BIM and BM25 IR models from scratch
* Apply them to the 20 Newsgroups corpus
* Score and rank documents based on a user query
* Visualize and explain document ranking

---

## Step-by-Step Explanation

### ① Document and Query Representation

Both documents and queries are represented as **vectors of terms**:

* Binary in BIM (term is present or not)
* Frequency-based in BM25 (how often a term occurs)

Let:

* Vocabulary: `["cat", "xylophone"]`
* Query: `q = ["cat", "xylophone"]`
* Document D3: `d = ["cat", "xylophone"]`

Then for D3:

```
TF: {"cat": 1, "xylophone": 1}, length = 4
```

---

### ② Binary Independence Model (BIM)

**Goal**: Rank documents by probability of relevance:

$$
P(R = 1 | d, q)
$$

#### ✲ Bayes Rule

Using Bayes:

$$
\text{Odds} = \frac{P(R=1 \mid \vec{x}, q)}{P(R=0 \mid \vec{x}, q)} = \frac{P(\vec{x} \mid R=1, q)}{P(\vec{x} \mid R=0, q)}
$$

#### ✲ Independence Assumption

$$
\frac{P(\vec{x} \mid R=1, q)}{P(\vec{x} \mid R=0, q)} = \prod_i \frac{P(x_i \mid R=1, q)}{P(x_i \mid R=0, q)}
$$

#### ✲ Final Score: Retrieval Status Value (RSV)

Let:

* $p_i = P(x_i=1 \mid R=1)$
* $u_i = P(x_i=1 \mid R=0)$

Then:

$$
RSV_d = \sum_{i:x_i=1, q_i=1} \log \left( \frac{p_i (1 - u_i)}{u_i (1 - p_i)} \right)
$$

Each term's contribution:

$$
c_i = \log \left( \frac{p_i}{1 - p_i} \cdot \frac{1 - u_i}{u_i} \right)
$$

#### Example BIM Calculation:

* `cat` and `xylophone` both appear in D3 and query
* Assume: $p_i = 0.5$, $u_i = 0.67$
* Then:

$$
c_i \approx \log(0.4925) \approx -0.307
$$

$$
RSV_{D3} = -0.307 + (-0.307) = -0.614
$$

---

### ③ BM25 Model

**Motivation**: BIM ignores term frequency and document length. BM25 addresses both.

#### ✲ BM25 Formula:

$$
RSV_d = \sum_{t \in q} idf_t \cdot \frac{(k_1 + 1) \cdot tf_{t,d}}{k_1 \cdot ((1 - b) + b \cdot \frac{L_d}{L_{avg}}) + tf_{t,d}}
$$

Where:

* $k_1$ controls term frequency scaling (typical: 1.5)
* $b$ controls document length normalization (typical: 0.75)
* $idf_t$ = $\log \left( \frac{N - df_t + 0.5}{df_t + 0.5} + 1 \right)$

#### Example BM25 for D3:

Let:

* $tf_{\text{cat}} = tf_{\text{xylophone}} = 1$
* $L_d = 4, L_{avg} = 3$
* $df = 2, N = 3$
* $idf \approx 0.47$

For each term:

$$
\text{score}_t = 0.47 \cdot \frac{2.5 \cdot 1}{1.875 + 1} \approx 0.409
$$

$$
RSV_{D3} = 0.409 + 0.409 = 0.818
$$

---

## 🚪 How to Run

1. Clone this repository:

```bash
git clone https://github.com/ahmedtarek26/Probalistic_IR.git
cd Probalistic_IR
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run  main file:

```bash
python main.py
```

---

## 🤖 Author

**Ahmed Tarek** 
DSAI master's student @UNITS
[LinkedIn](https://www.linkedin.com/in/ahmedtarek26)

**Ines Elgata** 
DSAI master's student @UNITS
[LinkedIn](https://www.linkedin.com/in/ines-el-gataa-b071aa206/)

---

## 🔖 License

MIT License

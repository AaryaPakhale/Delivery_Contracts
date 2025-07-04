# DeliveryContract Pricing System

This project implements a complete pricing engine in modern C++ for **DeliveryContracts**, a financial derivative that grants the right to deliver the most economical asset from a basket of **ValueNotes**. It addresses both foundational modeling and advanced pricing analytics, using three distinct rate conventions and a hybrid of analytic and simulation-based methods.

---

## 📌 Problem Overview

The challenge involves building a robust, scalable pricing system for:

1. **ValueNotes**: Tradable instruments involving principal and periodic interest, modeled under:
   - **Linear Rate Convention**
   - **Cumulative Rate Convention**
   - **Recursive Rate Convention**

2. **DeliveryContracts**: Derivatives where the seller delivers the *most economical* ValueNote from a basket, priced via:
   - **Forward modeling**
   - **Quadratic approximation**
   - **Monte Carlo simulation (bonus)**

---

## 📁 Project Structure

- **`Question2.cpp`**  
  Self-contained C++ file with all class definitions, logic, and output generation (CSV + console).

- **`submission_output.csv`**  
  Output file generated with numerical answers for all sub-questions Q1 and Q2.

- **`Pricing Problem Statement.pdf`**  
  Contains the full description of tasks, specifications, and submission format.

- **`documentation.docx`**  
  Contains the assumptions made, analytical derivations, methodology rationale, justifications and notes on extensibility, maintainability, and design patterns used.

---

## 🚀 Features Implemented

### ✅ Question 1: ValueNote Pricing

- **Q1.1** Price ↔ Effective Rate conversion under all 3 conventions:
  - Linear: closed-form
  - Cumulative: time-discounted cashflows
  - Recursive: forward value recursion

- **Q1.2** First Derivatives:
  - ∂VP/∂ER and ∂ER/∂VP (analytical, no finite differences)

- **Q1.3** Second Derivatives:
  - ∂²VP/∂ER² and ∂²ER/∂VP² (fully derived and implemented)

### ✅ Question 2: DeliveryContract Pricing

- **Q2.1** 
  - Basket construction of 4 ValueNotes
  - Calculation of Relative Factors using Cumulative method at Standardized Value Rate (SVR)

- **Q2.2** 
  - Pricing via expected min(Price/RelativeFactor) over z ∈ [-3, 3]
  - Fitted using **Weighted Quadratic Approximation** with normal distribution weights

- **Q2.3** 
  - Computation of delivery probability for each ValueNote (area under min region)

- **Q2.4**
  - Sensitivity of DeliveryContract price:
    - (a) to volatility of each ValueNote (∂Price/∂σᵢ)
    - (b) to today's price of each ValueNote (∂Price/∂VP₀ᵢ)
  - Fully derived using chain rule; avoids numerical shocks

- **Q2.5 [BONUS]**
  - **Monte Carlo Simulation** with Box-Muller normal generator
  - Approximates expected min(Price/RelativeFactor) over random samples

---

## 🛠️ Design Highlights

- **Modular OOP Design**
  - `ValueNote` encapsulates pricing and derivatives
  - `DeliveryContract` builds upon `ValueNote` for complex modeling
  - `QuadraticApproximation` abstracts polynomial fitting

- **Numerical Stability**
  - Newton-Raphson used with tolerances and fallbacks
  - Domain clamping and exception safety

- **Performance**
  - Avoids recalculations via caching z-points and derivatives
  - Uses std algorithms and manual memory control where helpful

---

## 📊 Output Format

All results are exported to `submission_output.csv` in the prescribed tabular format:

- **Q1:** For VN1
  - Prices and rates under all 3 conventions
  - First and second derivatives
- **Q2:** For all VNs
  - Relative factors
  - Delivery probabilities
  - Sensitivities to volatility and initial price
  - Total price of DeliveryContract

---

## 📎 How to Run

1. **Compile:**
   ```bash
   g++ -std=c++17 -O2 Question2.cpp -o pricing_model
   ```
2. **Run executable:**

    ```bash
    ./pricing_model
    ```

3. **Check output:**
    - `submission_output.csv` will be created in the current directory
    - Console will print comprehensive test results

---

## 📈 Dependencies

- Standard **C++17** libraries only
- No external dependencies or frameworks required

---

## 📚 Documentation

See the accompanying `documentation.docx` file for:

- Assumptions made
- Analytical derivations
- Methodology rationale and justifications
- Notes on extensibility, maintainability, and design patterns used

---

## 🏆 Bonus Notes

- Bonus pricing via **Monte Carlo simulation** introduces randomness and better models real-world optionality
- Quadratic fitting uses **normally weighted least squares** for better accuracy around mean values
- Validation includes probability normalization and Monte Carlo cross-checking

---

## 🔐 License

This project was developed as part of a quantitative pricing challenge and is intended for academic or evaluative purposes only.

---

## 👨‍💻 Author

**Aarya Pakhale**  
Finalist, Nomura Global Markets Quant Challenge 2025  

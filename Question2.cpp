#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <numeric>
#include <functional>
#include <ctime>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Quadratic approximation structure
struct QuadraticApproximation {
    double a, b, c;
    QuadraticApproximation(double a_, double b_, double c_) : a(a_), b(b_), c(c_) {}
    double evaluate(double z) const {
        return a * z * z + b * z + c;
    }
};

// Market data structure
struct MarketData {
    double riskFreeRate;
    double expirationTime; // in years
    double standardizedValueRate;
    
    MarketData(double rfr, double expTime, double svr) 
        : riskFreeRate(rfr), expirationTime(expTime), standardizedValueRate(svr) {}
};

class ValueNote {
private:
    double notional;        // N - Principal amount
    double maturity;        // M - Maturity in years
    double valueRate;       // VR - Annual interest rate (%)
    int paymentFrequency;   // PF - Number of payments per year
    double volatility;      // σ - Effective rate volatility (in %)
    double currentPrice;    // Current market price

public:
    // Helper function to calculate individual interest payment
    double calculateInterestPayment() const {
        return (valueRate * notional) / (100.0 * paymentFrequency);
    }
    
    // Helper function to get payment times
    std::vector<double> getPaymentTimes() const {
        std::vector<double> times;
        double timeStep = 1.0 / paymentFrequency;
        for (int i = 1; i <= static_cast<int>(maturity * paymentFrequency); ++i) {
            times.push_back(i * timeStep);
        }
        return times;
    }
public:
    // Constructor
    ValueNote(double n, double m, double vr, int pf, double vol = 0.0, double price = 0.0) 
        : notional(n), maturity(m), valueRate(vr), paymentFrequency(pf), 
          volatility(vol), currentPrice(price) {
        if (n <= 0 || m <= 0 || pf <= 0) {
            throw std::invalid_argument("Invalid ValueNote parameters");
        }
        if (vol < 0) {
            throw std::invalid_argument("Volatility cannot be negative");
        }
    }
    
    // Getters
    double getNotional() const { return notional; }
    double getMaturity() const { return maturity; }
    double getValueRate() const { return valueRate; }
    int getPaymentFrequency() const { return paymentFrequency; }
    double getVolatility() const { return volatility; }
    double getCurrentPrice() const { return currentPrice; }
    
    // Setters
    void setCurrentPrice(double price) { currentPrice = price; }
    void setVolatility(double vol) { volatility = vol; }
    
    // ============ LINEAR RATE CONVENTION ============
    
    // Q1.1a: Price given Effective Rate (Linear)
    double priceFromEffectiveRateLinear(double effectiveRate) const {
        return notional * (1.0 - effectiveRate * maturity / 100.0);
    }
    
    // Q1.1b: Effective Rate given Price (Linear)
    double effectiveRateFromPriceLinear(double price) const {
        return 100.0 * (notional - price) / (notional * maturity);
    }
    
    // Q1.2a: First derivative dVP/dER (Linear)
    double priceDerivativeLinear(double effectiveRate) const {
        return -notional * maturity / 100.0;
    }
    
    // Q1.2b: First derivative dER/dVP (Linear)
    double rateDerivativeLinear(double price) const {
        return -100.0 / (notional * maturity);
    }
    
    // Q1.3a: Second derivative d²VP/dER² (Linear)
    double priceSecondDerivativeLinear(double effectiveRate) const {
        return 0.0; // Linear function has zero second derivative
    }
    
    // Q1.3b: Second derivative d²ER/dVP² (Linear)
    double rateSecondDerivativeLinear(double price) const {
        return 0.0; // Linear function has zero second derivative
    }
    
    // ============ CUMULATIVE RATE CONVENTION ============
    
    // Q1.1a: Price given Effective Rate (Cumulative)
    double priceFromEffectiveRateCumulative(double effectiveRate) const {
        std::vector<double> paymentTimes = getPaymentTimes();
        double interestPayment = calculateInterestPayment();
        double price = 0.0;
        double discountRate = effectiveRate / (100.0 * paymentFrequency);
        
        // Sum discounted interest payments
        for (size_t i = 0; i < paymentTimes.size(); ++i) {
            double discountFactor = std::pow(1.0 + discountRate, paymentFrequency * paymentTimes[i]);
            price += interestPayment / discountFactor;
        }
        
        // Add discounted principal payment at maturity
        if (!paymentTimes.empty()) {
            double finalDiscountFactor = std::pow(1.0 + discountRate, paymentFrequency * maturity);
            price += notional / finalDiscountFactor;
        }
        
        return price;
    }
    
    // Q1.1b: Effective Rate given Price (Cumulative) - Newton-Raphson method
    double effectiveRateFromPriceCumulative(double price) const {
        const double tolerance = 1e-10;
        const int maxIterations = 100;
        double rate = 5.0; // Initial guess
        
        for (int iter = 0; iter < maxIterations; ++iter) {
            double f = priceFromEffectiveRateCumulative(rate) - price;
            double fprime = priceDerivativeCumulative(rate);
            
            if (std::abs(fprime) < tolerance) {
                throw std::runtime_error("Derivative too small in Newton-Raphson");
            }
            
            double newRate = rate - f / fprime;
            
            if (std::abs(newRate - rate) < tolerance) {
                return newRate;
            }
            
            rate = newRate;
        }
        
        throw std::runtime_error("Newton-Raphson failed to converge");
    }
    
    // Q1.2a: First derivative dVP/dER (Cumulative)
    double priceDerivativeCumulative(double effectiveRate) const {
        std::vector<double> paymentTimes = getPaymentTimes();
        double interestPayment = calculateInterestPayment();
        double derivative = 0.0;
        double discountRate = effectiveRate / (100.0 * paymentFrequency);
        
        // Derivative of discounted interest payments
        for (size_t i = 0; i < paymentTimes.size(); ++i) {
            double t = paymentTimes[i];
            double discountFactor = std::pow(1.0 + discountRate, paymentFrequency * t);
            derivative -= interestPayment * t / (100.0 * discountFactor * (1.0 + discountRate));
        }
        
        // Derivative of discounted principal
        if (!paymentTimes.empty()) {
            double discountFactor = std::pow(1.0 + discountRate, paymentFrequency * maturity);
            derivative -= notional * maturity / (100.0 * discountFactor * (1.0 + discountRate));
        }
        
        return derivative;
    }
    
    // Q1.2b: First derivative dER/dVP (Cumulative)
    double rateDerivativeCumulative(double price) const {
        double effectiveRate = effectiveRateFromPriceCumulative(price);
        double priceDerivative = priceDerivativeCumulative(effectiveRate);
        return 1.0 / priceDerivative;
    }
    
    // Q1.3a: Second derivative d²VP/dER² (Cumulative)
    double priceSecondDerivativeCumulative(double effectiveRate) const {
        std::vector<double> paymentTimes = getPaymentTimes();
        double interestPayment = calculateInterestPayment();
        double secondDerivative = 0.0;
        double discountRate = effectiveRate / (100.0 * paymentFrequency);
        
        // Second derivative of discounted interest payments
        for (size_t i = 0; i < paymentTimes.size(); ++i) {
            double t = paymentTimes[i];
            double discountFactor = std::pow(1.0 + discountRate, paymentFrequency * t);
            double term1 = interestPayment * t * t / (10000.0 * discountFactor * std::pow(1.0 + discountRate, 2));
            double term2 = interestPayment * (paymentFrequency * t + 1) * t / (10000.0 * paymentFrequency * discountFactor * std::pow(1.0 + discountRate, 2));
            secondDerivative += term1 + term2;
        }
        
        // Second derivative of discounted principal
        if (!paymentTimes.empty()) {
            double discountFactor = std::pow(1.0 + discountRate, paymentFrequency * maturity);
            double term1 = notional * maturity * maturity / (10000.0 * discountFactor * std::pow(1.0 + discountRate, 2));
            double term2 = notional * (paymentFrequency * maturity + 1) * maturity / (10000.0 * paymentFrequency * discountFactor * std::pow(1.0 + discountRate, 2));
            secondDerivative += term1 + term2;
        }
        
        return secondDerivative;
    }
    
    // Q1.3b: Second derivative d²ER/dVP² (Cumulative)
    double rateSecondDerivativeCumulative(double price) const {
        double effectiveRate = effectiveRateFromPriceCumulative(price);
        double firstDerivPrice = priceDerivativeCumulative(effectiveRate);
        double secondDerivPrice = priceSecondDerivativeCumulative(effectiveRate);
        
        return -secondDerivPrice / std::pow(firstDerivPrice, 3);
    }
    
    // ============ RECURSIVE RATE CONVENTION ============
    
    // Helper function to calculate FV_n for recursive method
    double calculateFVn(double effectiveRate) const {
        std::vector<double> paymentTimes = getPaymentTimes();
        double interestPayment = calculateInterestPayment();
        double fv = 0.0;
        
        for (size_t i = 0; i < paymentTimes.size(); ++i) {
            double timeDiff = (i == paymentTimes.size() - 1) ? 0.0 : (paymentTimes[i+1] - paymentTimes[i]);
            fv = (fv + interestPayment) * (1.0 + effectiveRate * timeDiff / 100.0);
        }
        
        return fv;
    }
    
    // Helper function to calculate derivative of FV_n
    double calculateFVnDerivative(double effectiveRate) const {
        std::vector<double> paymentTimes = getPaymentTimes();
        double interestPayment = calculateInterestPayment();
        double fv = 0.0;
        double fvDerivative = 0.0;
        
        for (size_t i = 0; i < paymentTimes.size(); ++i) {
            double timeDiff = (i == paymentTimes.size() - 1) ? 0.0 : (paymentTimes[i+1] - paymentTimes[i]);
            double growthFactor = 1.0 + effectiveRate * timeDiff / 100.0;
            
            double newFvDerivative = fvDerivative * growthFactor + (fv + interestPayment) * timeDiff / 100.0;
            fv = (fv + interestPayment) * growthFactor;
            fvDerivative = newFvDerivative;
        }
        
        return fvDerivative;
    }
    
    // Helper function to calculate second derivative of FV_n
    double calculateFVnSecondDerivative(double effectiveRate) const {
        std::vector<double> paymentTimes = getPaymentTimes();
        double interestPayment = calculateInterestPayment();
        double fv = 0.0;
        double fvDerivative = 0.0;
        double fvSecondDerivative = 0.0;
        
        for (size_t i = 0; i < paymentTimes.size(); ++i) {
            double timeDiff = (i == paymentTimes.size() - 1) ? 0.0 : (paymentTimes[i+1] - paymentTimes[i]);
            double growthFactor = 1.0 + effectiveRate * timeDiff / 100.0;
            
            double newFvSecondDerivative = fvSecondDerivative * growthFactor + 
                                         2.0 * fvDerivative * timeDiff / 100.0 + 
                                         (fv + interestPayment) * timeDiff * timeDiff / 10000.0;
            double newFvDerivative = fvDerivative * growthFactor + (fv + interestPayment) * timeDiff / 100.0;
            fv = (fv + interestPayment) * growthFactor;
            
            fvSecondDerivative = newFvSecondDerivative;
            fvDerivative = newFvDerivative;
        }
        
        return fvSecondDerivative;
    }
    
    // Q1.1a: Price given Effective Rate (Recursive)
    double priceFromEffectiveRateRecursive(double effectiveRate) const {
        double fvn = calculateFVn(effectiveRate);
        return (notional + fvn) / (1.0 + effectiveRate * maturity / 100.0);
    }
    
    // Q1.1b: Effective Rate given Price (Recursive) - Newton-Raphson method
    double effectiveRateFromPriceRecursive(double price) const {
        const double tolerance = 1e-10;
        const int maxIterations = 100;
        double rate = 5.0; // Initial guess
        
        for (int iter = 0; iter < maxIterations; ++iter) {
            double f = priceFromEffectiveRateRecursive(rate) - price;
            double fprime = priceDerivativeRecursive(rate);
            
            if (std::abs(fprime) < tolerance) {
                throw std::runtime_error("Derivative too small in Newton-Raphson");
            }
            
            double newRate = rate - f / fprime;
            
            if (std::abs(newRate - rate) < tolerance) {
                return newRate;
            }
            
            rate = newRate;
        }
        
        throw std::runtime_error("Newton-Raphson failed to converge");
    }
    
    // Q1.2a: First derivative dVP/dER (Recursive)
    double priceDerivativeRecursive(double effectiveRate) const {
        double fvn = calculateFVn(effectiveRate);
        double fvnDerivative = calculateFVnDerivative(effectiveRate);
        double denominator = 1.0 + effectiveRate * maturity / 100.0;
        
        double numeratorDerivative = fvnDerivative;
        double denominatorDerivative = maturity / 100.0;
        
        return (numeratorDerivative * denominator - (notional + fvn) * denominatorDerivative) / 
               (denominator * denominator);
    }
    
    // Q1.2b: First derivative dER/dVP (Recursive)
    double rateDerivativeRecursive(double price) const {
        double effectiveRate = effectiveRateFromPriceRecursive(price);
        double priceDerivative = priceDerivativeRecursive(effectiveRate);
        return 1.0 / priceDerivative;
    }
    
    // Q1.3a: Second derivative d²VP/dER² (Recursive)
    double priceSecondDerivativeRecursive(double effectiveRate) const {
        double fvn = calculateFVn(effectiveRate);
        double fvnDerivative = calculateFVnDerivative(effectiveRate);
        double fvnSecondDerivative = calculateFVnSecondDerivative(effectiveRate);
        double denominator = 1.0 + effectiveRate * maturity / 100.0;
        
        // Using quotient rule for second derivative: (u/v)'' = (u''v - uv''/v² - 2u'v'/v²)
        double u = notional + fvn;
        double uPrime = fvnDerivative;
        double uDoublePrime = fvnSecondDerivative;
        double v = denominator;
        double vPrime = maturity / 100.0;
        double vDoublePrime = 0.0;
        
        return (uDoublePrime * v - u * vDoublePrime) / (v * v) - 2.0 * uPrime * vPrime / (v * v);
    }
    
    // Q1.3b: Second derivative d²ER/dVP² (Recursive)
    double rateSecondDerivativeRecursive(double price) const {
        double effectiveRate = effectiveRateFromPriceRecursive(price);
        double firstDerivPrice = priceDerivativeRecursive(effectiveRate);
        double secondDerivPrice = priceSecondDerivativeRecursive(effectiveRate);
        
        return -secondDerivPrice / std::pow(firstDerivPrice, 3);
    }
};

// DeliveryContract class
class DeliveryContract {
private:
    std::vector<ValueNote> basket;
    std::vector<double> relativeFactors;
    MarketData marketData;
    std::vector<QuadraticApproximation> quadraticApproximations;
    std::vector<std::vector<double>> intersectionPoints;
    std::vector<std::pair<double, int>> deliveryRegions; // (z_point, most_economical_index)
    
    const int NUM_POINTS = 2000;
    const double Z_MIN = -3.0;
    const double Z_MAX = 3.0;
    
public:
    DeliveryContract(const std::vector<ValueNote>& vnBasket, const MarketData& mktData)
        : basket(vnBasket), marketData(mktData) {
        relativeFactors.resize(basket.size());
        calculateRelativeFactors();
        buildQuadraticApproximations();
        findDeliveryRegions();
    }
    
    // Q2.1a) Create basket of ValueNotes (already done in constructor)
    const std::vector<ValueNote>& getBasket() const { return basket; }
    
    // Q2.1b) Calculate RelativeFactor for each ValueNote
    void calculateRelativeFactors() {
        for (size_t i = 0; i < basket.size(); ++i) {
            // CumulativeFactor: RF = Price when ER = SVR (divided by 100)
            double priceAtSVR = basket[i].priceFromEffectiveRateCumulative(marketData.standardizedValueRate);
            relativeFactors[i] = priceAtSVR / 100.0;
        }
    }
    
    std::vector<double> getRelativeFactors() const { return relativeFactors; }
    
    // Calculate forward price
    double calculateForwardPrice(const ValueNote& vn, double timeToExpiration) const {
        std::vector<double> paymentTimes = vn.getPaymentTimes();
        double forwardPrice = (1.0 + marketData.riskFreeRate) * vn.getCurrentPrice();
        
        // Subtract present value of coupons paid before expiration
        double interestPayment = vn.calculateInterestPayment();
        for (double paymentTime : paymentTimes) {
            if (paymentTime <= timeToExpiration) {
                forwardPrice -= interestPayment * std::pow(1.0 + marketData.riskFreeRate, timeToExpiration - paymentTime);
            }
        }
        
        return forwardPrice;
    }
    
    // Solve quadratic equation for risk-adjusted effective rate
    double solveForRiskAdjustedRate(const ValueNote& vn, double forwardPrice) const {
        // Using Newton-Raphson to solve the complex quadratic equation
        const double tolerance = 1e-10;
        const int maxIterations = 100;
        double rate = 5.0; // Initial guess
        
        for (int iter = 0; iter < maxIterations; ++iter) {
            double price1 = vn.priceFromEffectiveRateCumulative(rate);
            double deriv1 = vn.priceDerivativeCumulative(rate);
            double deriv2 = vn.priceSecondDerivativeCumulative(rate);
            double vol = vn.getVolatility() / 100.0; // Convert to decimal
            double T = marketData.expirationTime;
            
            // The quadratic equation from the problem statement
            double f = 0.5 * deriv2 * std::exp(vol * vol * T) + 
                      rate * deriv1 - deriv2 * rate + 
                      0.5 * deriv2 * rate * rate - deriv1 * rate - forwardPrice;
            
            double fprime = deriv1 - deriv2 + deriv2 * rate - deriv1;
            
            if (std::abs(fprime) < tolerance) {
                break;
            }
            
            double newRate = rate - f / fprime;
            if (std::abs(newRate - rate) < tolerance) {
                return newRate;
            }
            rate = newRate;
        }
        
        // If Newton-Raphson fails, use approximation
        return vn.effectiveRateFromPriceCumulative(forwardPrice);
    }
    
    // Calculate effective rate for given z
    double calculateEffectiveRate(const ValueNote& vn, double z, double riskAdjustedRate) const {
        double vol = vn.getVolatility() / 100.0;
        double T = marketData.expirationTime;
        return riskAdjustedRate * std::exp(vol * z * std::sqrt(T) - 0.5 * vol * vol * T);
    }
    
    // Calculate price-to-relative-factor ratio
    double calculatePriceToRFRatio(size_t vnIndex, double z) const {
        double forwardPrice = calculateForwardPrice(basket[vnIndex], marketData.expirationTime);
        double riskAdjustedRate = solveForRiskAdjustedRate(basket[vnIndex], forwardPrice);
        double effectiveRate = calculateEffectiveRate(basket[vnIndex], z, riskAdjustedRate);
        double price = basket[vnIndex].priceFromEffectiveRateCumulative(effectiveRate);
        
        return price / relativeFactors[vnIndex];
    }
    
    // Build quadratic approximations
    void buildQuadraticApproximations() {
        quadraticApproximations.clear();
        
        for (size_t i = 0; i < basket.size(); ++i) {
            std::vector<double> zPoints(NUM_POINTS);
            std::vector<double> ratios(NUM_POINTS);
            std::vector<double> weights(NUM_POINTS);
            
            // Generate points and calculate ratios
            for (int j = 0; j < NUM_POINTS; ++j) {
                double z = Z_MIN + (Z_MAX - Z_MIN) * j / (NUM_POINTS - 1);
                zPoints[j] = z;
                ratios[j] = calculatePriceToRFRatio(i, z);
                
                // Normal weights (higher weight near mean)
                weights[j] = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
            }
            
            // Fit quadratic using weighted least squares
            QuadraticApproximation quad = fitWeightedQuadratic(zPoints, ratios, weights);
            quadraticApproximations.push_back(quad);
        }
    }
    
    // Weighted quadratic fitting
    QuadraticApproximation fitWeightedQuadratic(const std::vector<double>& x, 
                                               const std::vector<double>& y, 
                                               const std::vector<double>& w) const {
        // Weighted least squares for quadratic: y = ax² + bx + c
        double sum_w = 0, sum_wx = 0, sum_wx2 = 0, sum_wx3 = 0, sum_wx4 = 0;
        double sum_wy = 0, sum_wxy = 0, sum_wx2y = 0;
        
        for (size_t i = 0; i < x.size(); ++i) {
            double xi = x[i];
            double yi = y[i];
            double wi = w[i];
            
            sum_w += wi;
            sum_wx += wi * xi;
            sum_wx2 += wi * xi * xi;
            sum_wx3 += wi * xi * xi * xi;
            sum_wx4 += wi * xi * xi * xi * xi;
            sum_wy += wi * yi;
            sum_wxy += wi * xi * yi;
            sum_wx2y += wi * xi * xi * yi;
        }
        
        // Solve 3x3 system for a, b, c
        double det = sum_w * sum_wx2 * sum_wx4 + 2 * sum_wx * sum_wx2 * sum_wx3 - 
                    sum_wx2 * sum_wx2 * sum_wx2 - sum_w * sum_wx3 * sum_wx3 - 
                    sum_wx * sum_wx * sum_wx4;
        
        if (std::abs(det) < 1e-12) {
            // Fallback to simple quadratic
            return QuadraticApproximation(0.0, 0.0, y[NUM_POINTS/2]);
        }
        
        double a = (sum_wy * sum_wx2 * sum_wx4 + sum_wxy * sum_wx3 * sum_wx2 + 
                   sum_wx2y * sum_wx * sum_wx2 - sum_wx2y * sum_wx2 * sum_wx2 - 
                   sum_wy * sum_wx3 * sum_wx3 - sum_wxy * sum_wx * sum_wx4) / det;
        
        double b = (sum_w * sum_wxy * sum_wx4 + sum_wy * sum_wx3 * sum_wx2 + 
                   sum_wx2y * sum_wx * sum_wx - sum_wx2y * sum_w * sum_wx2 - 
                   sum_wy * sum_wx * sum_wx4 - sum_wxy * sum_wx3 * sum_wx) / det;
        
        double c = (sum_w * sum_wx2 * sum_wx2y + sum_wx * sum_wx3 * sum_wy + 
                   sum_wx2 * sum_wxy * sum_wx - sum_wx2 * sum_wx2 * sum_wy - 
                   sum_w * sum_wx3 * sum_wxy - sum_wx * sum_wx * sum_wx2y) / det;
        
        return QuadraticApproximation(a, b, c);
    }
    
    // Find intersection points and delivery regions
    void findDeliveryRegions() {
        intersectionPoints.clear();
        deliveryRegions.clear();
        
        std::vector<double> allIntersections;
        
        // Find all intersection points between pairs of quadratics
        for (size_t i = 0; i < quadraticApproximations.size(); ++i) {
            for (size_t j = i + 1; j < quadraticApproximations.size(); ++j) {
                std::vector<double> roots = findQuadraticIntersections(
                    quadraticApproximations[i], quadraticApproximations[j]);
                for (double root : roots) {
                    if (root >= Z_MIN && root <= Z_MAX) {
                        allIntersections.push_back(root);
                    }
                }
            }
        }
        
        // Sort intersection points
        allIntersections.push_back(Z_MIN);
        allIntersections.push_back(Z_MAX);
        std::sort(allIntersections.begin(), allIntersections.end());
        
        // Remove duplicates
        allIntersections.erase(std::unique(allIntersections.begin(), allIntersections.end(),
            [](double a, double b) { return std::abs(a - b) < 1e-8; }), allIntersections.end());
        
        // Determine most economical ValueNote in each region
        for (size_t i = 0; i < allIntersections.size() - 1; ++i) {
            double midPoint = (allIntersections[i] + allIntersections[i + 1]) / 2.0;
            
            int bestIndex = 0;
            double minRatio = quadraticApproximations[0].evaluate(midPoint);
            
            for (size_t j = 1; j < quadraticApproximations.size(); ++j) {
                double ratio = quadraticApproximations[j].evaluate(midPoint);
                if (ratio < minRatio) {
                    minRatio = ratio;
                    bestIndex = static_cast<int>(j);
                }
            }
            
            deliveryRegions.push_back({allIntersections[i + 1], bestIndex});
        }
    }
    
    // Find intersections between two quadratics
    std::vector<double> findQuadraticIntersections(const QuadraticApproximation& q1, 
                                                  const QuadraticApproximation& q2) const {
        std::vector<double> roots;
        
        // Solve (a1-a2)x² + (b1-b2)x + (c1-c2) = 0
        double a = q1.a - q2.a;
        double b = q1.b - q2.b;
        double c = q1.c - q2.c;
        
        if (std::abs(a) < 1e-12) {
            // Linear case
            if (std::abs(b) > 1e-12) {
                roots.push_back(-c / b);
            }
        } else {
            // Quadratic case
            double discriminant = b * b - 4 * a * c;
            if (discriminant >= 0) {
                double sqrtDisc = std::sqrt(discriminant);
                roots.push_back((-b + sqrtDisc) / (2 * a));
                roots.push_back((-b - sqrtDisc) / (2 * a));
            }
        }
        
        return roots;
    }
    
    // Q2.2) Calculate DeliveryContract price
    double calculateDeliveryContractPrice() const {
        const int numIntegrationPoints = 1000;
        double price = 0.0;
        double dz = (Z_MAX - Z_MIN) / numIntegrationPoints;
        
        for (int i = 0; i < numIntegrationPoints; ++i) {
            double z = Z_MIN + (i + 0.5) * dz;
            
            // Find which ValueNote is most economical at this z
            int bestIndex = 0;
            double minRatio = quadraticApproximations[0].evaluate(z);
            
            for (size_t j = 1; j < quadraticApproximations.size(); ++j) {
                double ratio = quadraticApproximations[j].evaluate(z);
                if (ratio < minRatio) {
                    minRatio = ratio;
                    bestIndex = static_cast<int>(j);
                }
            }
            
            // Normal distribution weight
            double weight = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
            price += minRatio * weight * dz;
        }
        
        return price;
    }
    
    // Q2.3) Calculate delivery probabilities
    std::vector<double> calculateDeliveryProbabilities() const {
        std::vector<double> probabilities(basket.size(), 0.0);
        const int numIntegrationPoints = 1000;
        double dz = (Z_MAX - Z_MIN) / numIntegrationPoints;
        
        for (int i = 0; i < numIntegrationPoints; ++i) {
            double z = Z_MIN + (i + 0.5) * dz;
            
            // Find which ValueNote is most economical at this z
            int bestIndex = 0;
            double minRatio = quadraticApproximations[0].evaluate(z);
            
            for (size_t j = 1; j < quadraticApproximations.size(); ++j) {
                double ratio = quadraticApproximations[j].evaluate(z);
                if (ratio < minRatio) {
                    minRatio = ratio;
                    bestIndex = static_cast<int>(j);
                }
            }
            
            // Normal distribution weight
            double weight = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
            probabilities[bestIndex] += weight * dz;
        }
        
        return probabilities;
    }
    
    // Q2.4a) Sensitivity to volatility
    double calculateVolatilitySensitivity(size_t vnIndex) const {
        const int numIntegrationPoints = 1000;
        double dz = (Z_MAX - Z_MIN) / numIntegrationPoints;
        double sensitivity = 0.0;
        
        const ValueNote& vn = basket[vnIndex];
        double sigma = vn.getVolatility() / 100.0;
        double T = marketData.expirationTime;
        double rf = relativeFactors[vnIndex];
        
        for (int i = 0; i < numIntegrationPoints; ++i) {
            double z = Z_MIN + (i + 0.5) * dz;

            // Determine if this VN is the most economical at z
            int bestIndex = 0;
            double minRatio = quadraticApproximations[0].evaluate(z);
            for (size_t j = 1; j < basket.size(); ++j) {
                double ratio = quadraticApproximations[j].evaluate(z);
                if (ratio < minRatio) {
                    minRatio = ratio;
                    bestIndex = static_cast<int>(j);
                }
            }

            if (bestIndex != static_cast<int>(vnIndex)) continue;

            double forwardPrice = calculateForwardPrice(vn, T);
            double ER_bar = solveForRiskAdjustedRate(vn, forwardPrice);
            double ER = ER_bar * std::exp(sigma * z * std::sqrt(T) - 0.5 * sigma * sigma * T);
            double dER_dSigma = ER * (z * std::sqrt(T) - sigma * T);
            double dVP_dER = vn.priceDerivativeCumulative(ER);

            double dPrice_dSigma = dVP_dER * dER_dSigma / rf;
            double weight = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);

            sensitivity += dPrice_dSigma * weight * dz;
        }

        return sensitivity;
    }
    
    // Q2.4b) Sensitivity to current price
    double calculatePriceSensitivity(size_t vnIndex) const {
        const int numIntegrationPoints = 1000;
        double dz = (Z_MAX - Z_MIN) / numIntegrationPoints;
        double sensitivity = 0.0;

        const ValueNote& vn = basket[vnIndex];
        double sigma = vn.getVolatility() / 100.0;
        double T = marketData.expirationTime;
        double rf = relativeFactors[vnIndex];

        for (int i = 0; i < numIntegrationPoints; ++i) {
            double z = Z_MIN + (i + 0.5) * dz;

            // Determine if this VN is the most economical at z
            int bestIndex = 0;
            double minRatio = quadraticApproximations[0].evaluate(z);
            for (size_t j = 1; j < basket.size(); ++j) {
                double ratio = quadraticApproximations[j].evaluate(z);
                if (ratio < minRatio) {
                    minRatio = ratio;
                    bestIndex = static_cast<int>(j);
                }
            }

            if (bestIndex != static_cast<int>(vnIndex)) continue;

            double forwardPrice = calculateForwardPrice(vn, T);
            double ER_bar = solveForRiskAdjustedRate(vn, forwardPrice);
            double ER = ER_bar * std::exp(sigma * z * std::sqrt(T) - 0.5 * sigma * sigma * T);

            // Chain rule: dVP/dVP_o = dVP/dER * dER_bar/dFP * dFP/dVP_o
            double dVP_dER = vn.priceDerivativeCumulative(ER);
            double dERbar_dFP = 1.0 / vn.priceDerivativeCumulative(ER_bar);
            double dFP_dVPo = (1.0 + marketData.riskFreeRate);

            double dVP_dVPo = dVP_dER * dERbar_dFP * dFP_dVPo / rf;

            double weight = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
            sensitivity += dVP_dVPo * weight * dz;
        }

        return sensitivity;
    }
    
    // Q2.5) BONUS: Alternative Pricing Method - Monte Carlo Simulation
    double calculateDeliveryContractPriceMonteCarlo(int numSimulations = 100000) const {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        double totalPayoff = 0.0;
        
        for (int sim = 0; sim < numSimulations; ++sim) {
            // Generate random z from standard normal distribution
            double u1 = static_cast<double>(std::rand()) / RAND_MAX;
            double u2 = static_cast<double>(std::rand()) / RAND_MAX;
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2); // Box-Muller
            
            // Clamp z to our range
            z = std::max(-3.0, std::min(3.0, z));
            
            // Find minimum price-to-RF ratio for this z
            double minRatio = std::numeric_limits<double>::max();
            for (size_t i = 0; i < basket.size(); ++i) {
                double ratio = calculatePriceToRFRatio(i, z);
                minRatio = std::min(minRatio, ratio);
            }
            
            totalPayoff += minRatio;
        }
        
        return totalPayoff / numSimulations;
    }
    
    // Alternative pricing with cubic spline interpolation (placeholder)
    double calculateDeliveryContractPriceCubicSpline() const {
        // For demonstration - would need full cubic spline implementation
        std::cout << "Cubic spline method would provide smoother interpolation" << std::endl;
        return calculateDeliveryContractPrice(); // Fallback to quadratic for now
    }
};

// Combined CSV output generation function
void generateCompleteCSVOutput() {
    std::ofstream file("submission_output.csv");
    file << std::fixed << std::setprecision(8);

    // Create test ValueNote (VN1) for Q1 calculations
    ValueNote vn1(100.0, 5.0, 3.5, 1, 1.5);  
    double testER = 5.0;
    double testPrice = 100.0;

    // Create basket of ValueNotes for Q2 calculations
    std::vector<ValueNote> basket = {
        {100.0, 5.0, 3.5, 1, 1.5, 95.0},    // VN1
        {100.0, 1.5, 2.0, 2, 2.5, 97.0},    // VN2
        {100.0, 4.5, 3.25, 1, 1.5, 99.0},   // VN3
        {100.0, 10.0, 8.0, 4, 5.0, 100.0}   // VN4
    };
    
    // Market data: SVR=5%, Expiration=3 months, Risk-free=4%
    MarketData marketData(0.04, 0.25, 5.0);
    
    // Create DeliveryContract
    DeliveryContract contract(basket, marketData);
    
    // Get Q2 results
    std::vector<double> relativeFactors = contract.getRelativeFactors();
    double contractPrice = contract.calculateDeliveryContractPrice();
    std::vector<double> deliveryProbs = contract.calculateDeliveryProbabilities();
    
    // Calculate sensitivities for specific test values
    std::vector<double> volSensitivities;
    std::vector<double> priceSensitivities;
    
    for (size_t i = 0; i < basket.size(); ++i) {
        volSensitivities.push_back(contract.calculateVolatilitySensitivity(i));
        priceSensitivities.push_back(contract.calculatePriceSensitivity(i));
    }

    // Header
    file << ",Linear,Cumulative,Recursive,Q 2.1,Q 2.2,Q 2.3,Q 2.4 a),Q 2.4 b)\n";

    // Q1.1a - Price for ER = 5%
    file << "Q 1.1 a)," << vn1.priceFromEffectiveRateLinear(testER) << ","
         << vn1.priceFromEffectiveRateCumulative(testER) << ","
         << vn1.priceFromEffectiveRateRecursive(testER) << ","
         << relativeFactors[0] << "," << contractPrice << ","
         << deliveryProbs[0] << "," << volSensitivities[0] << ","
         << priceSensitivities[0] << "\n";

    // Q1.1b - Rate for VP = 100
    file << "Q 1.1 b)," << vn1.effectiveRateFromPriceLinear(testPrice) << ","
         << vn1.effectiveRateFromPriceCumulative(testPrice) << ","
         << vn1.effectiveRateFromPriceRecursive(testPrice) << ","
         << relativeFactors[1] << ",," << deliveryProbs[1] << ","
         << volSensitivities[1] << "," << priceSensitivities[1] << "\n";

    // Q1.2a - First derivative dVP/dER at ER = 5%
    file << "Q 1.2 a)," << vn1.priceDerivativeLinear(testER) << ","
         << vn1.priceDerivativeCumulative(testER) << ","
         << vn1.priceDerivativeRecursive(testER) << ","
         << relativeFactors[2] << ",," << deliveryProbs[2] << ","
         << volSensitivities[2] << "," << priceSensitivities[2] << "\n";

    // Q1.2b - First derivative dER/dVP at VP = 100
    file << "Q 1.2 b)," << vn1.rateDerivativeLinear(testPrice) << ","
         << vn1.rateDerivativeCumulative(testPrice) << ","
         << vn1.rateDerivativeRecursive(testPrice) << ","
         << relativeFactors[3] << ",," << deliveryProbs[3] << ","
         << volSensitivities[3] << "," << priceSensitivities[3] << "\n";

    // Q1.3a - Second derivative d²VP/dER² at ER = 5%
    file << "Q 1.3 a)," << vn1.priceSecondDerivativeLinear(testER) << ","
         << vn1.priceSecondDerivativeCumulative(testER) << ","
         << vn1.priceSecondDerivativeRecursive(testER) << ",,,,\n";

    // Q1.3b - Second derivative d²ER/dVP² at VP = 100
    file << "Q 1.3 b)," << vn1.rateSecondDerivativeLinear(testPrice) << ","
         << vn1.rateSecondDerivativeCumulative(testPrice) << ","
         << vn1.rateSecondDerivativeRecursive(testPrice) << ",,,,\n";

    file.close();
    std::cout << "Complete results saved to submission_output.csv in correct format.\n";
}

// Comprehensive test function to display results
void runComprehensiveTest() {
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "=== Comprehensive DeliveryContract System Test ===" << std::endl;
    
    // Test Q1 - ValueNote calculations
    std::cout << "\n=== Q1 - ValueNote Tests ===" << std::endl;
    ValueNote vn1(100.0, 5.0, 3.5, 1, 1.5);
    double testER = 5.0;
    double testPrice = 100.0;
    
    std::cout << "Linear Convention:" << std::endl;
    std::cout << "  Price from ER(5%): " << vn1.priceFromEffectiveRateLinear(testER) << std::endl;
    std::cout << "  ER from Price(100): " << vn1.effectiveRateFromPriceLinear(testPrice) << std::endl;
    std::cout << "  dVP/dER: " << vn1.priceDerivativeLinear(testER) << std::endl;
    std::cout << "  dER/dVP: " << vn1.rateDerivativeLinear(testPrice) << std::endl;
    
    std::cout << "\nCumulative Convention:" << std::endl;
    std::cout << "  Price from ER(5%): " << vn1.priceFromEffectiveRateCumulative(testER) << std::endl;
    std::cout << "  ER from Price(100): " << vn1.effectiveRateFromPriceCumulative(testPrice) << std::endl;
    std::cout << "  dVP/dER: " << vn1.priceDerivativeCumulative(testER) << std::endl;
    std::cout << "  dER/dVP: " << vn1.rateDerivativeCumulative(testPrice) << std::endl;
    
    std::cout << "\nRecursive Convention:" << std::endl;
    std::cout << "  Price from ER(5%): " << vn1.priceFromEffectiveRateRecursive(testER) << std::endl;
    std::cout << "  ER from Price(100): " << vn1.effectiveRateFromPriceRecursive(testPrice) << std::endl;
    std::cout << "  dVP/dER: " << vn1.priceDerivativeRecursive(testER) << std::endl;
    std::cout << "  dER/dVP: " << vn1.rateDerivativeRecursive(testPrice) << std::endl;
    
    // Test Q2 - DeliveryContract calculations
    std::cout << "\n=== Q2 - DeliveryContract Tests ===" << std::endl;
    
    std::vector<ValueNote> basket = {
        {100.0, 5.0, 3.5, 1, 1.5, 95.0},    // VN1
        {100.0, 1.5, 2.0, 2, 2.5, 97.0},    // VN2
        {100.0, 4.5, 3.25, 1, 1.5, 99.0},   // VN3
        {100.0, 10.0, 8.0, 4, 5.0, 100.0}   // VN4
    };
    
    MarketData marketData(0.04, 0.25, 5.0);
    DeliveryContract contract(basket, marketData);
    
    // Q2.1 - Relative Factors
    std::vector<double> relativeFactors = contract.getRelativeFactors();
    std::cout << "\nQ2.1 - Relative Factors:" << std::endl;
    for (size_t i = 0; i < relativeFactors.size(); ++i) {
        std::cout << "  VN" << (i+1) << " RF: " << relativeFactors[i] << std::endl;
    }
    
    // Q2.2 - Contract Price
    double contractPrice = contract.calculateDeliveryContractPrice();
    std::cout << "\nQ2.2 - DeliveryContract Price: " << contractPrice << std::endl;
    
    // Q2.3 - Delivery Probabilities
    std::vector<double> deliveryProbs = contract.calculateDeliveryProbabilities();
    std::cout << "\nQ2.3 - Delivery Probabilities:" << std::endl;
    double totalProb = 0.0;
    for (size_t i = 0; i < deliveryProbs.size(); ++i) {
        std::cout << "  VN" << (i+1) << " Probability: " << deliveryProbs[i] << std::endl;
        totalProb += deliveryProbs[i];
    }
    std::cout << "  Total Probability: " << totalProb << std::endl;
    
    // Q2.4 - Sensitivities
    std::cout << "\nQ2.4a - Volatility Sensitivities:" << std::endl;
    for (size_t i = 0; i < basket.size(); ++i) {
        double volSens = contract.calculateVolatilitySensitivity(i);
        std::cout << "  VN" << (i+1) << " Vol Sensitivity: " << volSens << std::endl;
    }
    
    std::cout << "\nQ2.4b - Price Sensitivities:" << std::endl;
    for (size_t i = 0; i < basket.size(); ++i) {
        double priceSens = contract.calculatePriceSensitivity(i);
        std::cout << "  VN" << (i+1) << " Price Sensitivity: " << priceSens << std::endl;
    }
    
    // Q2.5 - Bonus: Alternative Methods
    std::cout << "\nQ2.5 - BONUS: Alternative Pricing Methods:" << std::endl;
    double mcPrice = contract.calculateDeliveryContractPriceMonteCarlo(50000);
    std::cout << "  Monte Carlo Price (50k sims): " << mcPrice << std::endl;
    std::cout << "  Quadratic Method Price: " << contractPrice << std::endl;
    std::cout << "  Difference: " << std::abs(mcPrice - contractPrice) << " (" 
              << (std::abs(mcPrice - contractPrice) / contractPrice * 100) << "%)" << std::endl;
    
    // Validation checks
    std::cout << "\n=== Validation Summary ===" << std::endl;
    int checks = 0, passed = 0;
    
    // Check 1: Total probability should be close to 1
    checks++;
    if (std::abs(totalProb - 1.0) < 0.01) {
        std::cout << "✓ Probability sum validation passed" << std::endl;
        passed++;
    } else {
        std::cout << "✗ Probability sum validation failed: " << totalProb << std::endl;
    }
    
    // Check 2: All probabilities should be non-negative
    checks++;
    bool probsValid = true;
    for (double prob : deliveryProbs) {
        if (prob < 0) probsValid = false;
    }
    if (probsValid) {
        std::cout << "✓ Non-negative probabilities validation passed" << std::endl;
        passed++;
    } else {
        std::cout << "✗ Non-negative probabilities validation failed" << std::endl;
    }
    
    // Check 3: Contract price should be positive
    checks++;
    if (contractPrice > 0) {
        std::cout << "✓ Positive contract price validation passed" << std::endl;
        passed++;
    } else {
        std::cout << "✗ Positive contract price validation failed" << std::endl;
    }
    
    // Check 4: Monte Carlo vs Quadratic agreement (within 10%)
    checks++;
    double priceDiff = std::abs(mcPrice - contractPrice) / contractPrice;
    if (priceDiff < 0.10) {
        std::cout << "✓ Monte Carlo vs Quadratic agreement validation passed" << std::endl;
        passed++;
    } else {
        std::cout << "✗ Monte Carlo vs Quadratic agreement validation failed" << std::endl;
    }
    
    std::cout << "\nValidation Results: " << passed << "/" << checks << " checks passed" << std::endl;
    
    if (passed == checks) {
        std::cout << "🎉 All validations passed! Implementation appears correct." << std::endl;
    } else {
        std::cout << "⚠️  Some validations failed. Please review implementation." << std::endl;
    }
}

int main() {
    try {
        std::cout << "Starting DeliveryContract Complete Implementation..." << std::endl;
        
        // // Uncomment to run comprehensive tests
        // runComprehensiveTest();
        
        // Generate final CSV output
        std::cout << "\nGenerating CSV output..." << std::endl;
        generateCompleteCSVOutput();
        
        std::cout << "\nExecution completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
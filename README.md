# Efficient Data Stream Anomaly Detection with Rolling Statistics Mechanism

You can access the website that I already create via this [link](https://impossible-odille-hanthedev-3edd50ec.koyeb.app/).
If the website didn't show the programs, you can see the demonstration via this [link](https://drive.google.com/file/d/1qD85m-Yb3m6tNjmyRN8TKccIFyU-0t91/view?usp=sharing).

Or you can clone this github and try it locally, step by step:

```
pip install -r requirements.txt
python app.py
```
If you want to try another algorithm, you can do it by following this step:

```
cd other_algorithm
```
Because the packages was from each file have different packages, you can run ```pip install <packages>```

After finished installing every dependencies, you can run each file (e.g. anomaly_detection_autoencoder.py)
```
python anomaly_detection_autoencoder.py
```
The figure will comes out

## The Reason I choose Rolling Statistics Algorithm

1. Simplicity and Ease of Implementation
    - Minimal Computational Overhead: Utilizes fundamental statistical measures—rolling mean and rolling standard deviation—allowing for swift computations suitable for real-time applications.
    - Straightforward Codebase: Requires fewer lines of code without the complexities inherent in machine learning models.
    - Ease of Understanding: The intuitive nature of statistical measures makes the system more transparent, facilitating easier debugging and modifications.

2. Real-Time Performance and Scalability
    - Low Latency: The method updates statistics incrementally with each new data point, ensuring that anomaly detection occurs almost instantaneously.
    - Resource Efficiency: Due to its lightweight nature, rolling statistics consume minimal memory and CPU resources, allowing the system to scale effortlessly with high-frequency data streams.
    - Deterministic Behavior: The algorithm's performance is predictable, ensuring consistent detection times regardless of data volume.

3. Adaptability to Data Trends and Seasonality
    - Dynamic Thresholds: By recalculating the mean and standard deviation over a moving window, the method adjusts to changes in data trends, ensuring that anomaly detection remains relevant over time.
    - Flexibility in Window Size: The ROLLING_WINDOW parameter can be tuned to capture short-term fluctuations or long-term trends, offering versatility in handling various data behaviors.

4. High Interpretability and Explainability
    - Transparent Decision Making: Since anomalies are determined based on clear statistical thresholds, stakeholders can easily comprehend and trust the detection mechanism.
    - Diagnostic Insights: The rolling mean and standard deviation provide contextual information, assisting in diagnosing underlying issues or trends leading to anomalies.

5. Lower Maintenance and Operational Overheads
    - No Need for Retraining: Rolling statistics require no periodic retraining as the data evolves, contrasting with algorithms like Isolation Forest or Autoencoders, which may need retraining to stay accurate.
    - Simplified Monitoring: The system's operational aspects are streamlined, reducing the need for specialized expertise to maintain or update the anomaly detection mechanism.

### Conclusion
Choosing Rolling Statistics as the core anomaly detection algorithm offers a balanced blend of simplicity, performance, interpretability, and adaptability, making it an ideal choice for real-time data streaming applications. While advanced machine learning algorithms bring robust detection capabilities, their complexity and resource demands may not be justifiable in scenarios where swift, transparent, and scalable solutions are paramount. Moreover, maintaining rolling statistics is straightforward, ensuring that the system remains reliable and efficient with minimal operational overhead.

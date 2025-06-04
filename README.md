# Agentic AI Use Case for Predictive Analytics with Animated Data Visualisation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Patria & Co. — Project Assignment Brief**  
> Evaluating Agentic AI capabilities for real-world predictive analytics with dynamic data visualization

## 🎯 Project Overview

This project demonstrates the development of an advanced, real-world Agentic AI use case for predictive analytics, integrated with animated data visualization to effectively communicate insights. The system showcases autonomous AI behavior including task planning, decision-making, learning from feedback, and self-adjustment based on prediction outputs.

## 📋 Table of Contents

- [Objective](#objective)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Technical Implementation](#technical-implementation)
- [Animated Visualizations](#animated-visualizations)
- [Deliverables](#deliverables)
- [Evaluation Criteria](#evaluation-criteria)
- [Submission Guidelines](#submission-guidelines)
- [Bonus Features](#bonus-features)
- [Support](#support)

## 🔍 Objective

Develop an advanced Agentic AI system that demonstrates:
- **Autonomous Decision-Making**: AI agents that can plan, execute, and adapt
- **Predictive Analytics**: Robust forecasting models for business-critical insights
- **Dynamic Visualization**: Animated data presentations that communicate complex insights effectively
- **Real-World Application**: Solutions relevant to business, development, health, environment, finance, or society

## ⭐ Key Features

### 🤖 Agentic AI Capabilities
- **Task Planning**: Autonomous goal decomposition and execution strategies
- **Decision-Making**: Context-aware choices based on data analysis
- **Learning from Feedback**: Adaptive behavior based on prediction performance
- **Self-Adjustment**: Dynamic model tuning and parameter optimization

### 📊 Predictive Analytics
- Time-series forecasting
- Churn prediction
- Stock/price prediction
- Anomaly detection
- Custom business metrics prediction

### 🎬 Animated Data Visualization
- Dynamic, engaging visual storytelling
- Real-time data updates
- Interactive dashboards
- Professional-grade animations

## 🛠 Requirements

### Technical Stack
- **Python**: 3.8+
- **Core Libraries**: pandas, numpy, scikit-learn, xgboost, prophet, statsmodels
- **AI/LLM**: transformers, langchain, openai, llama-index
- **Agent Frameworks**: autogen, crewAI, semantic-kernel, agenthub
- **Visualization**: plotly, matplotlib.animation, flourish, manim, three.js

### System Requirements
- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU support optional but recommended for large models

## 📁 Repository Structure

```
📁 agentic-ai-predictive-analytics/
├── 📁 data/
│   ├── sample_data.csv
│   ├── processed/
│   └── external/
├── 📁 src/
│   ├── agent_model.py          # Core agentic AI logic
│   ├── prediction_pipeline.py  # ML prediction pipeline
│   ├── animate_output.py       # Visualization animations
│   ├── utils/
│   │   ├── data_processor.py
│   │   ├── model_utils.py
│   │   └── visualization_helpers.py
│   └── config/
│       ├── model_config.yaml
│       └── agent_config.yaml
├── 📁 notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_development.ipynb
├── 📁 tests/
│   ├── test_agent.py
│   ├── test_prediction.py
│   └── test_visualization.py
├── 📁 assets/
│   ├── images/
│   └── animations/
├── 📁 deployment/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements-deploy.txt
├── requirements.txt
├── README.md
├── deployment_link.txt
├── usecase_summary.pdf
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
└── LICENSE
```

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/agentic-ai-predictive-analytics.git
cd agentic-ai-predictive-analytics
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

### 5. Data Setup
```bash
python src/utils/data_processor.py --setup
```

## 🚀 Usage

### Quick Start
```bash
# Run the complete pipeline
python src/prediction_pipeline.py

# Start the web application
streamlit run deployment/app.py

# Generate animations
python src/animate_output.py
```

### Advanced Configuration
```bash
# Custom model training
python src/agent_model.py --config config/custom_config.yaml

# Batch prediction with visualization
python src/prediction_pipeline.py --batch --animate
```

## 🔧 Technical Implementation

### Agent Architecture
```python
class PredictiveAgent:
    def __init__(self):
        self.planner = TaskPlanner()
        self.predictor = PredictionEngine()
        self.learner = FeedbackLearner()
        self.visualizer = AnimatedVisualizer()
    
    def autonomous_predict(self, data):
        # Task planning
        plan = self.planner.create_plan(data)
        
        # Execute predictions
        predictions = self.predictor.predict(data, plan)
        
        # Learn from results
        self.learner.update_from_feedback(predictions)
        
        # Generate visualizations
        return self.visualizer.animate(predictions)
```

### Prediction Pipeline
- **Data Ingestion**: Automated data collection and preprocessing
- **Feature Engineering**: Dynamic feature selection and creation
- **Model Selection**: Automated algorithm selection based on data characteristics
- **Hyperparameter Tuning**: Adaptive optimization using feedback loops
- **Validation**: Robust cross-validation and performance monitoring

### Animation Framework
```python
# Example animation generation
def create_prediction_animation(predictions, timestamps):
    fig = go.Figure()
    
    # Add animated traces
    for i, prediction in enumerate(predictions):
        fig.add_trace(go.Scatter(
            x=timestamps[:i+1],
            y=prediction[:i+1],
            mode='lines+markers',
            name=f'Prediction Step {i}'
        ))
    
    # Configure animation
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [play_button, pause_button]
        }]
    )
    
    return fig
```

## 🎬 Animated Visualizations

### Supported Animation Types
- **Time-series Forecasting**: Dynamic line charts showing prediction evolution
- **Confidence Intervals**: Animated uncertainty bands
- **Feature Importance**: Dynamic bar charts showing changing importance
- **Model Performance**: Real-time accuracy and error metrics
- **Agent Decision Flow**: Visualization of AI reasoning process

### Example Outputs
- [📊 Sample Time-series Animation](assets/animations/timeseries_forecast.gif)
- [📈 Performance Metrics Dashboard](assets/animations/performance_dashboard.gif)
- [🤖 Agent Decision Visualization](assets/animations/agent_decisions.gif)

## 📦 Deliverables

| Deliverable | Description | Status |
|-------------|-------------|---------|
| **GitHub Repository** | Structured, documented codebase with animation, agentic AI logic, predictive model, and visual output | ✅ |
| **Live App/Demo** | Working, shareable URL showcasing the result (Streamlit/Gradio/Flask) | 🔗 [Demo Link](deployment_link.txt) |
| **Executive Summary** | Professional 2-page PDF summarizing use case, approach, and innovation | 📄 [PDF](usecase_summary.pdf) |
| **Email Submission** | GitHub link + PDF summary to rizky@patriaco.id | 📧 Ready |

## 📊 Evaluation Criteria

| Criteria | Weight | Focus Areas |
|----------|--------|-------------|
| **Agentic AI Logic** | 30% | Clarity, novelty, autonomous behavior |
| **Predictive Accuracy** | 25% | Model robustness, validation methodology |
| **Visualization Quality** | 20% | Animation clarity, insight communication |
| **Code Structure** | 15% | Reproducibility, documentation, testing |
| **Executive Summary** | 10% | Professional presentation, business impact |

## 📤 Submission Guidelines

### Deadline
**[Insert deadline date here]**

### Submission Details
- **Email**: rizky@patriaco.id
- **Subject**: `Submission – Data & AI Fellow Assessment – [Your Full Name]`
- **Content**:
  - GitHub repository link
  - Live application/demo link
  - Executive summary PDF attachment

### Pre-Submission Checklist
- [ ] All code is properly documented
- [ ] Repository structure follows the specified format
- [ ] Live demo is accessible and functional
- [ ] Executive summary is professional and complete
- [ ] All requirements are met and tested

## 🏆 Bonus Features

Earn extra credit by implementing:

- **🔍 RAG-Enhanced Agents**: Integration with retrieval-augmented generation
- **☁️ Cloud Deployment**: Hugging Face Spaces, Vercel, or Docker deployment
- **🔄 CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **🌐 Bilingual Documentation**: English + Indonesian documentation
- **📱 Mobile Responsiveness**: Cross-platform compatibility
- **🔒 Security Features**: Authentication and data protection
- **📈 Advanced Analytics**: Real-time monitoring and alerting

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_agent.py -v
pytest tests/test_prediction.py -v
pytest tests/test_visualization.py -v

# Generate coverage report
pytest --cov=src tests/
```

## 📚 Documentation

### API Documentation
```bash
# Generate API docs
pdoc --html src/ --output-dir docs/

# Start documentation server
pdoc --http localhost:8080 src/
```

### Jupyter Notebooks
- [📓 Exploratory Data Analysis](notebooks/exploratory_analysis.ipynb)
- [🔬 Model Development](notebooks/model_development.ipynb)
- [📊 Visualization Examples](notebooks/visualization_examples.ipynb)

## 🔧 Configuration

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_KEY=your_hf_key

# Model Configuration
MODEL_TYPE=transformer
PREDICTION_HORIZON=30
ANIMATION_FPS=24

# Deployment
STREAMLIT_SERVER_PORT=8501
DEBUG_MODE=False
```

### Model Configuration
```yaml
# config/model_config.yaml
agent:
  type: "autonomous_predictor"
  planning_horizon: 10
  learning_rate: 0.001
  
prediction:
  algorithms: ["xgboost", "prophet", "lstm"]
  validation_method: "time_series_split"
  metrics: ["mae", "rmse", "mape"]
  
visualization:
  animation_duration: 5.0
  frame_rate: 30
  interactive: true
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Technical Support
- **Email**: business@patriaco.id
- **Website**: [www.patriaco.id](https://www.patriaco.id)

### Common Issues
- **Installation Problems**: Check Python version and virtual environment setup
- **API Key Errors**: Verify environment variables are set correctly
- **Animation Performance**: Reduce frame rate or data complexity for better performance
- **Deployment Issues**: Ensure all dependencies are included in requirements.txt

### FAQ
**Q: Can I use different visualization libraries?**
A: Yes, as long as they produce animated outputs and meet the quality requirements.

**Q: What if my model doesn't achieve high accuracy?**
A: Focus on the agentic behavior and learning capabilities. Document your approach and learnings.

**Q: Can I use synthetic data?**
A: Yes, synthetic data is acceptable if it's realistic and well-documented.

---

<div align="center">
*Remember: Every expert was once a beginner. Your programming journey is unique, and we're here to support you every step of the way.*

## 🌟 Support This Project
**Follow me on GitHub**: [![GitHub Follow](https://img.shields.io/github/followers/Harrypatria?style=social)](https://github.com/Harrypatria?tab=followers)
**Star this repository**: [![GitHub Star](https://img.shields.io/github/stars/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX?style=social)](https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX/stargazers)
**Connect on LinkedIn**: [![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harry-patria/)

Click the buttons above to show your support!

</div>

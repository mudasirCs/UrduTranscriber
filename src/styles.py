class CustomCSS:
    STYLES = """
        <style>
            /* Main Layout */
            .main-title {
                color: #FF4B4B;
                text-align: center;
                margin-bottom: 2rem;
                font-size: 2.5rem;
            }

            /* Progress Bars */
            .stProgress > div > div > div {
                background-color: #FF4B4B;
            }

            /* Buttons */
            .stButton>button {
                background-color: #FF4B4B;
                color: white;
                border-radius: 5px;
                padding: 0.5rem 1rem;
                border: none;
                width: 100%;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #FF6B6B;
                transform: translateY(-2px);
            }

            /* Status Boxes */
            .status-box {
                padding: 1rem;
                border-radius: 5px;
                background-color: #F0F2F6;
                margin: 1rem 0;
                border: 1px solid #E0E2E6;
            }

            /* Metrics Display */
            .metrics-container {
                display: flex;
                justify-content: space-between;
                margin: 1rem 0;
            }
            .metric-box {
                background-color: #F0F2F6;
                padding: 1rem;
                border-radius: 5px;
                text-align: center;
                flex: 1;
                margin: 0 0.5rem;
                border: 1px solid #E0E2E6;
            }

            /* Video Preview */
            .video-preview {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            /* Recent Files */
            .recent-file {
                padding: 0.5rem;
                margin: 0.5rem 0;
                border-radius: 5px;
                background-color: #F0F2F6;
                border: 1px solid #E0E2E6;
                transition: all 0.3s ease;
            }
            .recent-file:hover {
                background-color: #E8EAF0;
                transform: translateX(5px);
            }

            /* Model Info Box */
            .model-info {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
                border: 1px solid #e9ecef;
            }

            /* Status Indicators */
            .status-success {
                color: #28a745;
                font-weight: bold;
            }
            .status-error {
                color: #dc3545;
                font-weight: bold;
            }
            .status-warning {
                color: #ffc107;
                font-weight: bold;
            }
            .status-info {
                color: #17a2b8;
                font-weight: bold;
            }

            /* Tables */
            .styled-table {
                border-collapse: collapse;
                width: 100%;
                margin: 1rem 0;
                border-radius: 5px;
                overflow: hidden;
            }
            .styled-table th {
                background-color: #FF4B4B;
                color: white;
                padding: 0.75rem;
                text-align: left;
            }
            .styled-table td {
                padding: 0.75rem;
                border-bottom: 1px solid #E0E2E6;
            }
            .styled-table tr:nth-child(even) {
                background-color: #F8F9FA;
            }
            .styled-table tr:hover {
                background-color: #F0F2F6;
            }

            /* Loading Animation */
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .loading {
                animation: pulse 1.5s infinite;
            }

            /* Tooltips */
            .tooltip {
                position: relative;
                display: inline-block;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                background-color: #555;
                color: white;
                text-align: center;
                padding: 5px;
                border-radius: 6px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .metrics-container {
                    flex-direction: column;
                }
                .metric-box {
                    margin: 0.5rem 0;
                }
                .main-title {
                    font-size: 2rem;
                }
            }
        </style>
    """
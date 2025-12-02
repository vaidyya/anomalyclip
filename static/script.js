document.addEventListener("DOMContentLoaded", () => {
    const socket = new WebSocket("ws://localhost:8000/ws");

    const videoUpload = document.getElementById("video-upload");
    const submitButton = document.getElementById("submit-button");
    const videoPlayer = document.getElementById("video-player");
    const probabilityLabel = document.getElementById("probability-label");
    const topClasses = document.getElementById("top-classes");

    let anomalyChart, probabilitiesChart;

    submitButton.addEventListener("click", () => {
        const file = videoUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                videoPlayer.src = e.target.result;
                videoPlayer.play();
                socket.send(JSON.stringify({ video: e.target.result }));
            };
            reader.readAsDataURL(file);
        }
    });

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (!anomalyChart) {
            const anomalyCtx = document.getElementById("anomaly-chart").getContext("2d");
            anomalyChart = new Chart(anomalyCtx, {
                type: "line",
                data: {
                    labels: [],
                    datasets: [{
                        label: "Anomaly Score",
                        data: [],
                        borderColor: "red",
                        fill: false,
                    }],
                },
                options: {
                    plugins: {
                        title: {
                            display: true,
                            text: 'Anomaly Score p(A) Over Time'
                        }
                    },
                    scales: {
                        y: {
                            min: 0,
                            max: 1,
                        },
                    },
                },
            });

            const probabilitiesCtx = document.getElementById("probabilities-chart").getContext("2d");
            probabilitiesChart = new Chart(probabilitiesCtx, {
                type: "bar",
                data: {
                    labels: [],
                    datasets: [{
                        label: "Probability",
                        data: [],
                        backgroundColor: "blue",
                    }],
                },
                options: {
                    indexAxis: 'y',
                    plugins: {
                        title: {
                            display: true,
                            text: 'Class Probabilities p(c|A)'
                        }
                    },
                    scales: {
                        x: {
                            min: 0,
                            max: 1,
                        },
                    },
                },
            });
        }

        // Update anomaly score chart
        anomalyChart.data.labels.push(anomalyChart.data.labels.length);
        anomalyChart.data.datasets[0].data.push(data.score);
        anomalyChart.update();

        // Update probabilities chart
        probabilitiesChart.data.labels = data.class_probabilities.labels;
        probabilitiesChart.data.datasets[0].data = data.class_probabilities.data;
        probabilitiesChart.update();

        // Update probability and top classes
        probabilityLabel.innerText = `p(A) = ${data.prob.toFixed(3)}`;
        let topClassesHtml = "<h3>Top anomalous classes:</h3><ol>";
        for (const cls of data.top_classes) {
            topClassesHtml += `<li>${cls.name} (${cls.prob.toFixed(3)})</li>`;
        }
        topClassesHtml += "</ol>";
        topClasses.innerHTML = topClassesHtml;
    };
});
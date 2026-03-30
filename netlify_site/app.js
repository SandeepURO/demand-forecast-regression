const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");
const today = new Date().toISOString().slice(0, 10);
document.getElementById("date").value = today;

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const apiUrl = document.getElementById("api-url").value.trim();
  const warehouse = Number(document.getElementById("warehouse").value);
  const category = Number(document.getElementById("category").value);
  const product = Number(document.getElementById("product").value);
  const date = document.getElementById("date").value;

  resultBox.classList.remove("hidden");
  resultBox.innerHTML = "<p>Predicting...</p>";

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ warehouse, category, product, date }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    resultBox.innerHTML = `
      <h2>Prediction Result</h2>
      <p><strong>Predicted Demand:</strong> ${data.predicted_demand} units</p>
      <p><strong>Latency:</strong> ${data.latency_ms} ms</p>
    `;
  } catch (error) {
    resultBox.innerHTML = `
      <h2>Error</h2>
      <p>${error.message}</p>
      <p>Check that your backend URL ends with <strong>/api/predict</strong>.</p>
    `;
  }
});

const $ = (id) => document.getElementById(id);
const plot = $("plot"), go = $("go"), errBox = $("error"), result = $("result");
const SAMPLE = "A young farm boy discovers he is the last heir of a fallen kingdom. With a rebel band of thieves and a wandering sorcerer, he crosses haunted forests and frozen mountains to find an ancient sword and defeat the dark king who destroyed his family.";

const wordCount = () => plot.value.trim().split(/\s+/).filter(Boolean).length;
plot.addEventListener("input", () => { $("count").textContent = `${wordCount()} words`; });
$("sample").addEventListener("click", () => { plot.value = SAMPLE; plot.dispatchEvent(new Event("input")); });

function showError(msg) { errBox.textContent = msg; errBox.hidden = false; result.hidden = true; }

go.addEventListener("click", async () => {
  errBox.hidden = true;
  if (wordCount() < 10) return showError("Write at least 10 words so the model has enough to work with.");
  go.disabled = true; go.textContent = "Predicting…";
  try {
    const res = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plot: plot.value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    render(data);
  } catch (e) {
    showError(e instanceof TypeError ? "Can't reach the server. Check that the Flask app is running." : e.message);
  } finally { go.disabled = false; go.textContent = "Predict genre"; }
});

function render(d) {
  $("genre").textContent = d.genre;
  $("conf").textContent = d.confidence == null ? "Confidence isn't available for this model" : `${(d.confidence * 100).toFixed(1)}% confident`;
  const bars = $("bars"); bars.replaceChildren();
  for (const p of d.top_predictions) {
    const li = document.createElement("li");
    li.innerHTML = '<span class="g"></span><span class="track"><span class="fill"></span></span><span class="p"></span>';
    li.querySelector(".g").textContent = p.genre;
    li.querySelector(".fill").style.width = `${p.confidence * 100}%`;
    li.querySelector(".p").textContent = `${(p.confidence * 100).toFixed(0)}%`;
    bars.append(li);
  }
  $("model").textContent = `Model: ${d.model}`;
  result.hidden = false;
}

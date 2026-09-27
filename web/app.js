let selected, slug, previewUrl;
let ready = false, busy = false;
let pairingVersion = 0;
function clearPairing() {
  pairingVersion++;
  el('pairing-details').hidden = true;
  el('pairing-result').textContent = '';
  el('pair').disabled = false;
}
const el=id=>document.getElementById(id);
const updateScan = () => { el('scan').disabled = !selected || !ready || busy || !navigator.onLine; };
async function checkHealth() {
  ready = false;
  updateScan();
  if (!navigator.onLine) {
    el('message').textContent = 'Нет сети. Интерфейс доступен, но для распознавания нужно подключение к серверу.';
    return;
  }
  try {
    const response = await fetch('/health', {cache: 'no-store', signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error();
    ready = (await response.json()).ready === true;
    if (!busy) el('message').textContent = ready ? 'Готово к распознаванию. Выберите фото этикетки.' : 'Просмотр интерфейса: распознавание отключено, пока не подключена модель.';
  } catch {
    if (!busy) el('message').textContent = 'Сервер недоступен. Проверьте подключение и повторите попытку.';
  }
  updateScan();
}
el('retry').addEventListener('click', checkHealth);
window.addEventListener('online', checkHealth);
window.addEventListener('offline', checkHealth);
checkHealth();
el('image').addEventListener('change', event => {
  selected = event.target.files[0];
  slug = null;
  clearPairing();
  el('result').hidden = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  el('preview').hidden = true;
  if (selected && selected.size > 20 * 1024 * 1024) {
    selected = null;
    el('image').value = '';
    el('message').textContent = 'Фотография больше 20 МБ. Выберите файл поменьше.';
  }
  if (selected) {
    previewUrl = URL.createObjectURL(selected);
    el('preview').src = previewUrl;
    el('preview').hidden = false;
  }
  updateScan();
});
el('scan').addEventListener('click',async()=>{if(!selected||!ready||busy||!navigator.onLine)return;busy=true;el('image').disabled=true;updateScan();el('message').textContent='Анализируем этикетку…';el('result').hidden=true;const body=new FormData();body.append('image',selected);try{const response=await fetch('/v1/predict',{method:'POST',body,signal:AbortSignal.timeout(60000)});if(!response.ok)throw new Error((await response.json()).detail||'Ошибка запроса');const data=await response.json();slug=data.slug;clearPairing();const card=data.card;el('state').textContent=data.status==='low_confidence_reshoot'?'Недостаточно уверенности — лучше переснять':'Найден кандидат';el('state').className=data.status==='low_confidence_reshoot'?'state low':'state';el('title').textContent=card['Название вина']||slug;el('winery').textContent=card['Винодельня']||'—';el('region').textContent=card['Регион']||'—';el('grape').textContent=card['Сорт винограда']||'—';el('category').textContent=card['Категория']||'—';el('description').textContent=card['Описание']||'';el('diagnostics').textContent=`Статус: ${data.status}. Время: ${Math.round(data.latency_ms)} мс. Сходство — технический сигнал, а не вероятность. ${data.ocr_text?'OCR: '+data.ocr_text:''}`;el('result').hidden=false;el('message').textContent=''}catch(error){el('message').textContent=error.message}finally{busy=false;el('image').disabled=false;updateScan()}});
el('dish').addEventListener('input', clearPairing);
el('dish').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !el('pair').disabled) el('pair').click();
});
function showList(id, items) {
  el(id).replaceChildren(...items.map(text => {
    const item = document.createElement('li');
    item.textContent = text;
    return item;
  }));
}
el('pair').addEventListener('click', async () => {
  if (!slug) return;
  clearPairing();
  const version = pairingVersion;
  const dish = el('dish').value.trim();
  if (!dish) {
    el('pairing-result').textContent = 'Укажите блюдо, например: рыба в сливочном соусе.';
    return;
  }
  const wineSlug = slug;
  el('pair').disabled = true;
  el('pairing-result').textContent = 'Подбираем сочетание…';
  try {
    const response = await fetch('/v1/pairing', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({slug: wineSlug, dish}), signal: AbortSignal.timeout(35000)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Не удалось получить рекомендацию. Проверьте введённое блюдо.');
    if (version !== pairingVersion || slug !== wineSlug) return;
    el('pairing-result').textContent = [data.provider_notice, data.explanation, data.question].filter(Boolean).join(' ');
    const hasScore = Number.isFinite(data.score);
    el('pairing-score').hidden = !hasScore;
    el('pairing-meter').hidden = !hasScore;
    if (hasScore) {
      el('pairing-score').textContent = `${data.score} / 100 · ${data.method === 'yandexgpt' ? 'оценка ИИ' : 'оценка по правилам'}`;
      el('pairing-meter').value = data.score;
    }
    showList('pairing-reasons', data.reasons || []);
    showList('pairing-tips', data.recommendations || []);
    el('pairing-disclaimer').textContent = data.disclaimer;
    el('pairing-details').hidden = !hasScore && !data.recommendations?.length;
  } catch (error) {
    if (version !== pairingVersion) return;
    el('pairing-result').textContent = navigator.onLine ? error.message : 'Для рекомендации нужно подключение к серверу.';
  } finally {
    if (version === pairingVersion) el('pair').disabled = false;
  }
});

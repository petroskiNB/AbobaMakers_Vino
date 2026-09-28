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
function scrollToText(id) {
  el(id).scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start'});
}
let healthVersion = 0;
async function checkHealth(manual = false) {
  const version = ++healthVersion;
  ready = false;
  updateScan();
  el('retry').disabled = true;
  el('retry').textContent = 'Проверяем…';
  el('connection-status').textContent = 'Проверяем подключение к серверу…';
  if (manual) scrollToText('connection-status');
  try {
    if (!navigator.onLine) throw new Error('offline');
    const response = await fetch('/health', {cache: 'no-store', signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error();
    const data = await response.json();
    if (version !== healthVersion) return;
    ready = data.ready === true;
    const time = new Date().toLocaleTimeString('ru-RU');
    el('connection-status').textContent = ready
      ? `Подключение проверено в ${time}. Сервер доступен, модель готова к распознаванию.`
      : `Подключение проверено в ${time}. Сервер доступен, но модель ещё не готова.`;
  } catch {
    if (version !== healthVersion) return;
    el('connection-status').textContent = navigator.onLine
      ? 'Не удалось проверить подключение. Сервер не ответил вовремя или произошла ошибка сети. Повторите проверку.'
      : 'Нет сети. Для распознавания нужно подключение к интернету.';
  } finally {
    if (version === healthVersion) {
      el('retry').disabled = false;
      el('retry').textContent = 'Проверить подключение';
      updateScan();
    }
  }
}
el('retry').addEventListener('click', () => checkHealth(true));
window.addEventListener('online', () => checkHealth());
window.addEventListener('offline', () => checkHealth());
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
function selectCandidate(candidate, rank) {
  slug = candidate.slug;
  clearPairing();
  const card = candidate.card || {};
  el('title').textContent = card['Название вина'] || slug;
  el('winery').textContent = card['Винодельня'] || '—';
  el('region').textContent = card['Регион'] || '—';
  el('grape').textContent = card['Сорт винограда'] || '—';
  el('category').textContent = card['Категория'] || '—';
  el('description').textContent = card['Описание'] || '';
  el('selection-note').textContent = rank === 1
    ? 'Открыт первый кандидат. Подбор блюда относится к этой карточке.'
    : `Вы выбрали кандидата №${rank}. Подбор блюда относится к выбранной карточке; выбор не подтверждает распознавание.`;
  el('candidates').querySelectorAll('button').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.slug === slug));
  });
}

function showCandidates(data) {
  const candidates = data.candidates?.length
    ? data.candidates.slice(0, 5)
    : [{slug: data.slug, card: data.card}];
  // Keep compatibility with a server that supplies only the winning card.
  candidates.forEach(candidate => {
    if (!candidate.card && candidate.slug === data.slug) candidate.card = data.card;
  });
  el('candidates').replaceChildren(...candidates.map((candidate, index) => {
    const item = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'candidate';
    button.dataset.slug = candidate.slug;
    button.setAttribute('aria-pressed', 'false');
    button.disabled = !candidate.card;
    const title = document.createElement('span');
    title.textContent = `${index + 1}. ${candidate.card?.['Название вина'] || candidate.slug}`;
    const subtitle = document.createElement('small');
    subtitle.textContent = [candidate.card?.['Винодельня'], candidate.card?.['Регион'],
      candidate.card?.['Категория']].filter(Boolean).join(' · ');
    button.append(title, subtitle);
    button.addEventListener('click', () => selectCandidate(candidate, index + 1));
    item.append(button);
    return item;
  }));
  el('candidates-panel').open = false;
  selectCandidate(candidates[0], 1);
}

el('scan').addEventListener('click', async () => {
  if (!selected || !ready || busy || !navigator.onLine) return;
  busy = true;
  slug = null;
  clearPairing();
  el('image').disabled = true;
  updateScan();
  el('message').textContent = 'Анализируем этикетку…';
  el('result').hidden = true;
  const body = new FormData();
  scrollToText('message');
  body.append('image', selected);
  try {
    const response = await fetch('/v1/predict', {method: 'POST', body, signal: AbortSignal.timeout(60000)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Ошибка запроса');
    const lowConfidence = data.status === 'low_confidence_reshoot';
    el('state').textContent = lowConfidence ? 'Недостаточно уверенности — лучше переснять' : 'Найден кандидат';
    el('state').className = lowConfidence ? 'state low' : 'state';
    showCandidates(data);
    el('diagnostics').textContent = `Статус поиска: ${data.status}. Время: ${Math.round(data.latency_ms)} мс. Сходство — технический сигнал, а не вероятность. ${data.ocr_text ? 'OCR: ' + data.ocr_text : ''}`;
    el('result').hidden = false;
    el('message').textContent = '';
    scrollToText('result');
  } catch (error) {
    el('message').textContent = error.message;
    scrollToText('message');
  } finally {
    busy = false;
    el('image').disabled = false;
    updateScan();
  }
});
el('dish').addEventListener('input', clearPairing);
el('dish').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !el('pair').disabled) el('pair').click();
});
function setAdviceText(target, text) {
  target.replaceChildren();
  // Only support bold markers. All model output remains text, never HTML.
  for (const part of String(text || '').split(/(\*\*[^*]+\*\*)/g)) {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      const strong = document.createElement('strong');
      strong.textContent = part.slice(2, -2);
      target.append(strong);
    } else {
      target.append(document.createTextNode(part));
    }
  }
}
function showList(id, items) {
  const entries = (Array.isArray(items) ? items : [])
    .filter(text => typeof text === 'string' && text.replace(/[\s*•\-–—]/g, '').length);
  el(id).hidden = entries.length === 0;
  el(id).replaceChildren(...entries.map(text => {
    const item = document.createElement('li');
    setAdviceText(item, text);
    return item;
  }));
  return entries.length;
}
el('pair').addEventListener('click', async () => {
  if (!slug) return;
  clearPairing();
  const version = pairingVersion;
  const dish = el('dish').value.trim();
  if (!dish) {
    el('pairing-result').textContent = 'Укажите блюдо, например: рыба в сливочном соусе.';
    scrollToText('pairing-result');
    return;
  }
  const wineSlug = slug;
  el('pair').disabled = true;
  el('pairing-result').textContent = 'Подбираем сочетание…';
  el('dish').blur();
  scrollToText('pairing-result');
  try {
    const response = await fetch('/v1/pairing', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({slug: wineSlug, dish}), signal: AbortSignal.timeout(35000)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Не удалось получить рекомендацию. Проверьте введённое блюдо.');
    if (version !== pairingVersion || slug !== wineSlug) return;
    setAdviceText(el('pairing-result'), [data.provider_notice, data.explanation, data.question].filter(Boolean).join(' '));
    const hasScore = Number.isFinite(data.score);
    el('pairing-score').hidden = !hasScore;
    el('pairing-meter').hidden = !hasScore;
    if (hasScore) {
      el('pairing-score').textContent = `${data.score} / 100 · ${data.method === 'yandexgpt' ? 'оценка ИИ' : 'оценка по правилам'}`;
      el('pairing-meter').value = data.score;
    }
    const reasonsCount = showList('pairing-reasons', data.reasons || []);
    const tipsCount = showList('pairing-tips', data.recommendations || []);
    el('pairing-tips-title').hidden = tipsCount === 0;
    el('pairing-disclaimer').textContent = data.disclaimer;
    el('pairing-details').hidden = !hasScore && !tipsCount && !reasonsCount;
    scrollToText('pairing-result');
  } catch (error) {
    if (version !== pairingVersion) return;
    el('pairing-result').textContent = navigator.onLine ? error.message : 'Для рекомендации нужно подключение к серверу.';
    scrollToText('pairing-result');
  } finally {
    if (version === pairingVersion) el('pair').disabled = false;
  }
});

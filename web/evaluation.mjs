export function metrics(rows, labels) {
  if (!rows.length) return null;
  const times = rows.map(r => r.latency_ms).sort((a,b)=>a-b);
  const result = {queries: rows.length, successful: rows.filter(r=>r.ok).length,
    mean_ms: times.reduce((a,b)=>a+b,0)/times.length,
    p95_ms: times[Math.ceil(.95*times.length)-1],
    successful_under_3s: rows.filter(r=>r.ok && r.latency_ms<3000).length, quality: null};
  if (labels) {
    let hits1=0, hits5=0, returned1=0, returned5=0;
    for (const r of rows) {
      const expected=labels.get(r.query_id);
      if (!expected) throw new Error('Нет разметки для '+r.query_id);
      const candidates=r.ok ? [...new Set(r.top5)].slice(0,5) : [];
      const first=r.ok ? r.predicted_slug : null;
      hits1+=first===expected; hits5+=candidates.includes(expected);
      returned1+=Boolean(first); returned5+=candidates.length;
    }
    result.quality={accuracy_top1:hits1/rows.length, micro_f1_top1:2*hits1/(returned1+rows.length),
      micro_f1_top5:2*hits5/(returned5+rows.length), recall_top5:hits5/rows.length};
  }
  return result;
}

export function parseTSV(text, fields) {
  const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/);
  const headers=lines.shift().split('\t');
  if (!fields.every(f=>headers.includes(f))) throw new Error('Нужны колонки: '+fields.join(', '));
  const seen=new Set();
  return lines.map(line=>{
    const values=line.split('\t');
    const row=Object.fromEntries(headers.map((h,i)=>[h,values[i]?.trim()]));
    if (!fields.every(f=>row[f]) || seen.has(row.query_id)) throw new Error('Пустые поля или повтор query_id');
    seen.add(row.query_id); return row;
  });
}

if (typeof document !== 'undefined') {
  const el=id=>document.getElementById(id);
  const pct=value=>(value*100).toFixed(2)+'%';
  let report;
  el('start').onclick=async()=>{
    el('start').disabled=true; el('download').disabled=true; report=null;
    el('rows').replaceChildren(); el('summary').textContent='';
    const rows=[];
    const origin=el('origin').value, originText=el('origin').selectedOptions[0].text;
    try {
      if (!el('manifest').files[0]) throw new Error('Выберите манифест');
      const queries=parseTSV(await el('manifest').files[0].text(),['query_id','image_path']);
      const files=new Map();
      for (const f of el('images').files) {
        if(files.has(f.name)) throw new Error('Повтор имени файла: '+f.name);
        files.set(f.name,f);
      }
      for(const q of queries) if(!files.has(q.image_path)) throw new Error('Не выбран файл '+q.image_path);
      let labels=null;
      if(el('labels').files[0]) {
        const items=parseTSV(await el('labels').files[0].text(),['query_id','expected_slug']);
        labels=new Map(items.map(r=>[r.query_id,r.expected_slug]));
        if(labels.size!==queries.length || queries.some(q=>!labels.has(q.query_id))) throw new Error('Разметка должна соответствовать всем запросам');
      }
      for(const q of queries) {
        el('status').textContent=`Обработка ${rows.length+1}/${queries.length}: ${q.image_path}`;
        const bytes=await files.get(q.image_path).arrayBuffer();
        const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
        const row={...q,image_sha256:hash,ok:false,predicted_slug:null,top5:[]};
        const body=new FormData();body.append('image',files.get(q.image_path));
        const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),10000);
        const started=performance.now();
        try {
          const response=await fetch('/v1/predict',{method:'POST',body,signal:controller.signal});
          row.http_status=response.status;
          if(!response.ok) throw new Error('HTTP '+response.status);
          const data=await response.json();
          const candidates=[...new Set((data.candidates||[]).map(c=>c.slug).filter(s=>typeof s==='string' && s))].slice(0,5);
          if(!data.slug || candidates[0]!==data.slug) throw new Error('Некорректные кандидаты');
          Object.assign(row,{ok:true,predicted_slug:data.slug,top5:candidates});
        } catch(e) {row.error=e.message;} finally {clearTimeout(timer);row.latency_ms=performance.now()-started;}
        rows.push(row);
        const tr=document.createElement('tr');
        for(const value of [q.query_id,(row.latency_ms/1000).toFixed(3),row.predicted_slug||row.error,row.top5.join('\n')]) {
          const td=document.createElement('td');td.textContent=value;tr.append(td);
        }
        el('rows').append(tr);
        const m=metrics(rows,labels);
        el('summary').textContent=`Ответы: ${m.successful}/${m.queries}. Среднее: ${(m.mean_ms/1000).toFixed(3)} с. p95: ${(m.p95_ms/1000).toFixed(3)} с. Быстрее 3 с: ${m.successful_under_3s}/${m.queries}. `+
          (m.quality?`F1 топ-1: ${pct(m.quality.micro_f1_top1)}; F1 топ-5: ${pct(m.quality.micro_f1_top5)}; Accuracy: ${pct(m.quality.accuracy_top1)}; Recall@5: ${pct(m.quality.recall_top5)}. Источник: ${originText}.`:'F1 и Accuracy не вычислены: нет разметки.');
        report={created_at:new Date().toISOString(),label_source:labels?origin:null,summary:m,rows};
        el('download').disabled=false;
      }
      el('status').textContent='Прогон завершён.';
    } catch(e) {el('status').textContent=e.message;} finally {el('start').disabled=false;}
  };
  el('download').onclick=()=>{
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='wine-evaluation.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  fetch('/static/synthetic-evaluation.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error();return r.json();}).then(r=>{
    const q=r.quality;
    el('synthetic').textContent=`${r.queries} запросов. Accuracy: ${pct(q.accuracy_top1)}; micro-F1 топ-1: ${pct(q.micro_set_f1_at_1)}; micro-F1 топ-5: ${pct(q.micro_set_f1_at_5)}; Recall@5: ${pct(q.recall_at_5)}. Не полевая точность.`;
  }).catch(()=>{el('synthetic').textContent='Сохранённый результат пока не опубликован.';});
}

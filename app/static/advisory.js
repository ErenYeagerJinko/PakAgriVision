/* Render only the server's confidence-gated advisory; never invent a fallback treatment. */
function renderTreatmentAdvisory(result, lang) {
  const advisory = result.advisory;
  const ur = lang === 'ur';
  const treatment = advisory.treatment;
  const text = (id, value) => { document.getElementById(id).textContent = value; };
  const banner = document.getElementById('primaryBanner');
  banner.className = 'primary-diagnosis-banner ' +
    (advisory.status === 'healthy' ? 'severity-healthy' : 'severity-moderate');
  const status = advisory.status === 'uncertain' ? (ur ? 'تصدیق ضروری ہے' : 'VERIFY SYMPTOMS') :
    advisory.status === 'healthy' ? (ur ? 'بظاہر تندرست' : 'PREDICTED HEALTHY') :
    (ur ? 'بیماری کا امکان' : 'DISEASE PREDICTED');
  text('resSeverityText', status);
  text('resSeverityIcon', advisory.status === 'healthy' ? '✓' : '⚠');
  document.getElementById('lowConfidenceAlert').classList.toggle('visible', advisory.low_confidence);
  text('advisoryWarningText', ur ? advisory.warning_ur : advisory.warning);
  text('treatmentImmediateText', treatment.immediate_action);
  text('treatmentChemicalText', treatment.chemical_treatment);
  text('treatmentOrganicText', treatment.organic_treatment);
  text('treatmentPreventionText', treatment.prevention);
  text('treatmentSeverityText', treatment.severity_guidance);
  text('advisoryDisclaimer', advisory.disclaimer);
  text('dosageNotice', advisory.dosage_notice);
  text('knowledgeLanguageNotice', ur ?
    'حفاظتی انتباہ اردو میں ہے؛ اصل حوالہ شدہ زرعی ہدایات انگریزی میں برقرار رکھی گئی ہیں۔' :
    'Source-backed guidance is retained in its reviewed English wording. Confidence is not disease severity.');
  text('treatmentReview', (ur ? 'آخری جائزہ: ' : 'Last reviewed: ') + treatment.last_reviewed);
  text('treatmentSourcesTitle', ur ? 'حوالہ جات' : 'Sources');
  const list = document.getElementById('treatmentSources');
  list.replaceChildren();
  for (const source of treatment.sources) {
    const li = document.createElement('li');
    const match = source.match(/https?:\/\/[^\s,]+/);
    if (match) {
      const link = document.createElement('a');
      link.href = match[0];
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = source;
      li.append(link);
    } else {
      li.textContent = source;
    }
    list.append(li);
  }
}

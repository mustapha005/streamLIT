import re
from collections import Counter
from pathlib import Path
import pandas as pd
import streamlit as st
from src.pipeline import EOS, NGramModel, InterpolatedModel, load_documents, split_by_document, backoff_top_k, top_k_accuracy
from src.evaluate import backoff_accuracy

st.set_page_config(page_title='Smart Keyboard', page_icon='⌨️', layout='wide')
st.markdown('''<style>.block-container {max-width:1120px;padding-top:2.5rem;} h1 {letter-spacing:-0.04em;} [data-testid="stMetric"] {background:#f0f4f8;border-radius:14px;padding:18px;color:#142c43;}</style>''', unsafe_allow_html=True)

@st.cache_data
def corpus():
    return load_documents(Path(__file__).parent / 'data' / 'corpus.csv')

@st.cache_resource
def models(k, training_only):
    docs = corpus()
    train, _ = split_by_document(docs)
    sentences = train if training_only else [s for d in docs.values() for s in d]
    return [NGramModel(sentences, n, k) for n in (3, 2, 1)], InterpolatedModel(sentences, k), sentences

@st.cache_data
def evaluate(k):
    train, test = split_by_document(corpus())
    ms = [NGramModel(train, n, k) for n in (3, 2, 1)]
    rows = []
    for name, model in list(zip(('Trigram', 'Bigram', 'Unigram'), ms)) + [('Interpolated', InterpolatedModel(train, k))]:
        rows.append({'Model': name, 'Perplexity': model.perplexity(test)[0], 'Top-3 accuracy': top_k_accuracy(model, test)[0]})
    rows.append({'Model': 'Hard back-off', 'Perplexity': None, 'Top-3 accuracy': backoff_accuracy(*ms, test)})
    return pd.DataFrame(rows), len(train), len(test)

def set_text(value):
    st.session_state.composer = value

def insert_word(word):
    text = st.session_state.composer.rstrip()
    if word == EOS:
        st.session_state.composer = text + ('' if text.endswith('.') else '.') + ' '
    elif word == '.':
        st.session_state.composer = text + '. '
    else:
        st.session_state.composer = text + (' ' if text else '') + word + ' '

with st.sidebar:
    st.title('⌨️ Keyboard lab')
    st.caption('LAB 02 · N-GRAM LANGUAGE MODELS')
    mode = st.selectbox('Prediction method', ['Hard back-off', 'Interpolated', 'Trigram', 'Bigram', 'Unigram'])
    smoothing = st.select_slider('Add-k smoothing', options=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0], value=0.1)
    count = st.slider('Number of suggestions', 3, 6, 3)
    training_only = st.toggle('Use training documents only', value=True)
    st.divider()
    st.caption('Hard back-off tries a trigram, then a bigram, then a unigram when the context is unseen.')
    st.caption('This small synthetic English corpus is a teaching dataset. Suggestions are limited to its vocabulary.')

(tri, bi, uni), interp, sentences = models(smoothing, training_only)
st.caption('LANGUAGE MODEL PLAYGROUND')
st.title('A little help with your next word.')
st.write('Compose a sentence, explore predictions, and see how context shapes each suggestion.')
compose, evaluation, dataset = st.tabs(['✍️ Compose', '📊 Model evaluation', '📚 Corpus explorer'])
with compose:
    st.session_state.setdefault('composer', 'the happy girls')
    st.text_area('Your sentence', key='composer', height=140, placeholder='Start with the girls…', help='Press Ctrl+Enter or leave this field to update predictions.')
    raw = st.session_state.composer
    tokens = re.findall(r"[\w]+(?:['’-][\w]+)*|[^\w\s]", raw.lower())
    if '.' in tokens:
        # A completed sentence starts a fresh prediction context.
        last = max(i for i, token in enumerate(tokens) if token == '.')
        tokens = tokens[last + 1:]
    unknown = sorted(set(w for w in tokens if w not in uni.vocab))
    if unknown:
        st.warning('Outside the training vocabulary: ' + ', '.join(unknown) + '. Treated as <unk>.')
    if mode == 'Hard back-off':
        suggestions, used = backoff_top_k(tri, bi, uni, tokens, count)
    else:
        selected = {'Interpolated': interp, 'Trigram': tri, 'Bigram': bi, 'Unigram': uni}[mode]
        suggestions, used = selected.top_k(tokens, count), mode.lower()
    st.subheader('Your next word')
    st.caption(f'Using {used} · click a suggestion to insert it')
    for col, (word, probability) in zip(st.columns(count), suggestions):
        with col:
            st.button('Finish sentence' if word == EOS else word, key=f'suggest_{word}', on_click=insert_word, args=(word,), width='stretch', type='primary')
            st.caption(f'{probability:.1%} model probability')
    st.caption('Probabilities use the full vocabulary; displayed suggestions need not sum to 100%.')
    st.divider()
    a, b, c, d = st.columns(4)
    a.button('Try: the girls', on_click=set_text, args=('the girls',), width='stretch')
    b.button('Try: in the', on_click=set_text, args=('in the',), width='stretch')
    c.button('New sentence', on_click=set_text, args=('',), width='stretch')
    d.download_button('Save text', raw, 'my-sentence.txt', 'text/plain', width='stretch')
    with st.expander('What is the model seeing?'):
        st.write('Trigram context:', ' → '.join(tri.context(tokens)))
        st.write('Bigram context:', ' → '.join(bi.context(tokens)))
        st.write('Trigram context observed:', tri.seen_context(tokens))
        st.write('Bigram context observed:', bi.seen_context(tokens))
with evaluation:
    st.subheader('How well does it predict unseen documents?')
    st.write('Evaluation always trains on the first 80% of documents and tests on the last 20%, independently of the composer setting.')
    st.caption('Perplexity: lower is better. Top-3 accuracy: higher is better. Back-off measures suggestion accuracy only.')
    if st.button('Run evaluation', type='primary'):
        with st.spinner('Scoring held-out sentences…'):
            frame, n_train, n_test = evaluate(smoothing)
        st.session_state.evaluation = (smoothing, frame, n_train, n_test)
    if 'evaluation' in st.session_state:
        evaluated_k, frame, n_train, n_test = st.session_state.evaluation
        st.caption(f'Results for k={evaluated_k} · {n_train:,} training sentences · {n_test:,} test sentences')
        if evaluated_k != smoothing:
            st.info('Smoothing changed. Run evaluation again to refresh results.')
        st.dataframe(frame, hide_index=True, width='stretch', column_config={'Top-3 accuracy': st.column_config.NumberColumn(format='percent'), 'Perplexity': st.column_config.NumberColumn(format='%.2f')})
        st.bar_chart(frame.set_index('Model')[['Top-3 accuracy']], color='#2563eb')
        st.download_button('Download evaluation CSV', frame.to_csv(index=False), 'evaluation.csv', 'text/csv')
with dataset:
    docs = corpus()
    all_sentences = [s for d in docs.values() for s in d]
    a, b, c = st.columns(3)
    a.metric('Documents', len(docs))
    b.metric('Sentences', f'{len(all_sentences):,}')
    c.metric('Composer vocabulary', len(uni.vocab - {EOS, '<unk>'}))
    st.caption(f'The composer currently uses {len(sentences):,} sentences.')
    query = st.text_input('Find a sentence', placeholder='e.g. kitchen')
    rows = [{'Document': doc, 'Sentence': ' '.join(s)} for doc, ss in docs.items() for s in ss if query.casefold() in ' '.join(s).casefold()]
    st.dataframe(pd.DataFrame(rows, columns=['Document', 'Sentence']), hide_index=True, width='stretch', height=280)
    frequent = Counter(w for s in sentences for w in s if w != '.')
    st.subheader('Most frequent words in the composer corpus')
    st.bar_chart(pd.DataFrame(frequent.most_common(15), columns=['Word', 'Count']).set_index('Word'))


from risk_qa.contracts import EvidencePassage


def passage(pid="p1", text="Operational risk controls", page=1, authority="OSFI", release="r1"):
    return EvidencePassage(passage_id=pid, doc_id="doc", version_id="v1", authority=authority,
                           pdf_page_index=page, text=text, char_start=0, char_end=len(text),
                           pdf_url="https://example.org/source.pdf", release_id=release)

import spacy
import re
import chardet

# Load SpaCy model (English only)
nlp = spacy.load("en_core_web_sm")

def clean_email(text):
    """
    Clean email text by removing subject line and normalizing whitespace.
    """
    # Ensure UTF-8 encoding
    try:
        if isinstance(text, bytes):
            encoding = chardet.detect(text)["encoding"]
            text = text.decode(encoding or "utf-8", errors="ignore")
        else:
            text = text.encode("latin1").decode("utf-8", errors="ignore")
    except:
        pass
    
    # Normalize newlines to ensure consistent processing
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove subject line
    if "Subject:" in text:
        # Try splitting on double newline
        parts = text.split("\n\n", 1)
        if len(parts) > 1:
            text = parts[1]
        else:
            # Fallback: search for salutations
            salutations = ["Sehr geehrte", "Dear ", "Hallo "]
            for sal in salutations:
                if sal in text:
                    text = text[text.index(sal):]
                    break
            else:
                # Take after "Subject:"
                text = text.split("Subject:", 1)[1].strip()
    
    # Normalize whitespace after processing
    text = " ".join(text.split())
    return text


def mask_pii(email_text):
    """
    Mask PII in email text using SpaCy for dates and regex for other entities.
    Args:
        email_text (str): Input email text
    Returns:
        tuple: (masked_email, list_of_entities)
    """
    # Clean email first
    cleaned_text = clean_email(email_text)
    entities = []
    masked_text = cleaned_text

    # Process with SpaCy for entity recognition
    doc = nlp(cleaned_text)
    
    # Extract and mask dates first
    date_entities = []
    for ent in doc.ents:
        if ent.label_ == "DATE" and not re.match(r"\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}|\d{4}\s\d{4}\s\d{4}", ent.text):
            # Only consider date formats that look like dates (not card numbers)
            if re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{1,2}[- ]\w+[- ]\d{2,4}", ent.text):
                date_entities.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": "dob"
                })
    
    # Sort date entities by position (descending) to avoid offset issues
    for date_entity in sorted(date_entities, key=lambda x: x["start"], reverse=True):
        entity_value = date_entity["text"]
        start, end = date_entity["start"], date_entity["end"]
        entities.append({
            "position": [start, end],
            "classification": "dob",
            "entity": entity_value
        })
        masked_text = masked_text[:start] + "[dob]" + masked_text[end:]

    # Enhanced regex patterns for PII with context capturing - with improved full name detection
    patterns = [
        # Full name patterns with context and improved specificity
        (r"((?:My name is|name[:]+|My full name is) )([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})(?=\s|$|\.|\,)", "[full_name]", "full_name", True),
        (r"(?:^|\. |, )([A-Z][a-z]+ [A-Z][a-z]+)(?=$|\. |\,)", "[full_name]", "full_name", False),
        
        # Email patterns - modified to more precisely capture just the email with context
        (r"((?:email(?:[ ]?(?:me|is|at))?:? |reach me at |contact(?:[ ]?(?:me|at))?:? ))(\S+@\S+\.\S+)(?=[,. ]|$)", "[email]", "email", True),
        (r"(\S+@\S+\.\S+)(?=[,. ]|$)", "[email]", "email", False),
        
        # Phone number patterns (international format support)
        # (r"((?:phone(?:[ ]?(?:is|at|number))?:? |call me at ))?\b((?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4})\b", "[phone_number]", "phone_number", True),
        (r"((?:phone(?:[ ]?(?:is|at|number))?|contact(?: number)?(?: is)?|call me at )[: ]*)?\b((?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4})\b", "[phone_number]", "phone_number", True),

        # Identification numbers with potential context
        (r"((?:Aadhar(?:[ ]?(?:is|number))?:? ))?(\b\d{4}\s\d{4}\s\d{4}\b)", "[aadhar_num]", "aadhar_num", True),
        (r"((?:card(?:[ ]?(?:is|number))?:? ))?(\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b)", "[credit_debit_no]", "credit_debit_no", True),
        
        # Security codes with context
        (r"((?:CVV:? ))(\d{3})\b", "[cvv_no]", "cvv_no", True),
        
        # Expiry dates with context
        (r"((?:expiry:? ))(\d{2}/\d{2})\b", "[expiry_no]", "expiry_no", True),
        (r"\b(0[1-9]|1[0-2])/([0-9]{2})\b", "[expiry_no]", "expiry_no", False)  # More specific expiry format MM/YY
    ]

    # Apply regex patterns sequentially
    for pattern, replacement, entity_type, has_context in patterns:
        matches = list(re.finditer(pattern, masked_text, re.IGNORECASE))
        
        # Sort matches by position (descending) to avoid offset issues when replacing
        for match in sorted(matches, key=lambda x: x.start(), reverse=True):
            start, end = match.start(), match.end()
            
            # Extract the actual entity value based on whether pattern has context or not
            if has_context:
                context = match.group(1)
                entity_value = match.group(2)
                context_start = start
                context_end = start + len(context) if context else start
                entity_start = context_end
                entity_end = end
            else:
                context = ""
                entity_value = match.group(1)
                context_start = start
                context_end = start
                entity_start = start
                entity_end = end
            
            # Skip if this region was already masked
            if any(entity_start >= e["position"][0] and entity_end <= e["position"][1] for e in entities):
                continue
                
            # Skip if the match contains a placeholder from previous replacements
            if re.search(r"\[\w+\]", masked_text[entity_start:entity_end]):
                continue
            
            # Store both entity and context information
            entities.append({
                "position": [entity_start, entity_end],
                "classification": entity_type,
                "entity": entity_value,
                "context": context.strip() if context else "",
                "context_position": [context_start, context_end]
            })
            
            # Preserve context in masked text if it exists
            if context:
                masked_text = masked_text[:context_end] + replacement + masked_text[entity_end:]
            else:
                masked_text = masked_text[:entity_start] + replacement + masked_text[entity_end:]

    # Remove any added context entries from the final entities list
    for i in range(len(entities)):
        if "context" in entities[i]:
            del entities[i]["context"]
        if "context_position" in entities[i]:
            del entities[i]["context_position"]

    return masked_text, entities

def demask_email(masked_email, entities):
    """
    Restore original email from masked email using stored entities while preserving context phrases.
    
    Args:
        masked_email (str): The masked email text with placeholders
        entities (list): List of entity dictionaries with position, classification, and entity
        
    Returns:
        str: Email with original entities restored while preserving context phrases
    """
    # Sort entities by position (ascending) for proper replacement
    sorted_entities = sorted(entities, key=lambda x: x["position"][0])
    
    # Create a result by working from the end to the beginning to avoid offset issues
    demasked_email = masked_email
    
    # Create a mapping of placeholders to their entities
    placeholder_map = {}
    for entity in sorted_entities:
        classification = entity["classification"]
        placeholder = f"[{classification}]"
        
        # Group entities by classification
        if placeholder not in placeholder_map:
            placeholder_map[placeholder] = []
        
        # Include context information if available
        entity_info = {
            "entity": entity["entity"],
            "context": entity.get("context", "")
        }
        
        placeholder_map[placeholder].append(entity_info)
    
    # Replace placeholders with original values while preserving context
    for placeholder, entity_infos in placeholder_map.items():
        # Count how many instances of this placeholder exist
        placeholder_count = demasked_email.count(placeholder)
        
        # If we have fewer entities than placeholders, reuse the last entity
        if placeholder_count > len(entity_infos):
            entity_infos += [entity_infos[-1]] * (placeholder_count - len(entity_infos))
        
        # Replace each instance of the placeholder
        for i in range(min(placeholder_count, len(entity_infos))):
            entity_info = entity_infos[i]
            entity_value = entity_info["entity"]
            
            # Replace the placeholder with the entity value
            demasked_email = demasked_email.replace(placeholder, entity_value, 1)
    
    return demasked_email
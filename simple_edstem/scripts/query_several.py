import parse_utils

class QuerySeveral:
    word_freq_dict = {} # the dictionary of words to dict of ids to counts
    id_title_dict = {}  # the dictionary of ids to titles

    
    def __init__(self, wikifile):
        parse_utils.parse(wikifile, self.process_page)


    def process_page(self, wiki_page:str):
        """
        reads one wiki/xml file, processes each page, populating the
        dictionary that maps words->page_id->frequency counts
        
        parameters:
            wiki_page   the path to an xml file with pages (each with title, 
                        id, and text sections)
        """

        page_id = int(wiki_page.find("id").text)
        page_title = wiki_page.find("title").text.strip()
        page_text = wiki_page.find("text").text.strip()
        if wiki_page.find("text").text is None: page_text = ""
        
        self.id_title_dict[page_id] = page_title

        tokens = parse_utils.get_tokens(page_title + " " + page_text)
        
        for word in tokens:
            word = parse_utils.stem_and_stop(word)
            if word != "":
                if parse_utils.word_is_link(word):
                    # split link into the text and the destination, but 
                    # only process the text
                    # include link text because it's part of the page text
                    link_text = parse_utils.split_link(word)[1]
                    for text in link_text:
                        if text in self.word_freq_dict:
                            if page_id in self.word_freq_dict[text]:
                                self.word_freq_dict[text][page_id] += 1
                            else: 
                                self.word_freq_dict[text][page_id] = 1
                        else: 
                            self.word_freq_dict[text] = {page_id : 1}

                else:
                    # for non-links, just record its presence
                    if word in self.word_freq_dict:
                        if page_id in self.word_freq_dict[word]:
                            self.word_freq_dict[word][page_id] += 1
                        else: 
                            self.word_freq_dict[word][page_id] = 1
                    else: 
                        self.word_freq_dict[word] = {page_id : 1}

    def query(self, search_term:str, format="title") -> list:
        """
        searches for page titles that contain the search term

        Parameters:
        search_term -- the string to search for in wiki pages; for this
                       assignment these can be just single words

        format -- used to control whether a list of page ids or titles 
                  is returned. title is the default, but the value can
                  be set to "id" when query is called to get the page 
                  ids instead (ids might be less error-prone to check in tests)
        
        Returns:
        the list of pages that contain the search term (as per the format)
        """
        if format not in ["id", "title"]:
            raise ValueError("Invalid results format " + format)
        
        term_low = search_term.lower()
        if term_low in parse_utils.STOP_WORDS:
            print("WARNING: STOP WORD isn't indexed -- " + search_term)
            return []

        term_low = parse_utils.stem_and_stop(term_low)
        if term_low in self.word_freq_dict:
            # if the term is in the dictionary, get the list of ids and sort it
            sorted_ids = sorted(self.word_freq_dict[term_low], 
                            key = self.word_freq_dict[term_low].__getitem__,
                            reverse = True)
            
            # If our query is for ids, just return the sorted list of ids
            if format=="id": 
                return sorted_ids
            
            # If our query is for titles, convert the sorted list of ids to titles
            elif format=="title": 
                titles = []
                for id in sorted_ids:
                    titles.append(self.id_title_dict[id])
                return titles
            
        else: # term not in dictionary
            return []
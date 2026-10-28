"""Porter stemmer (Porter 1980), pure Python."""
import re


class PorterStemmer:
    V = "[aeiou]|[^aeiou]y"  # treated via helper functions below

    def _cons(self, w, i):
        c = w[i]
        if c in "aeiou":
            return False
        if c == "y":
            return i == 0 or not self._cons(w, i - 1)
        return True

    def _m(self, w):
        n, i, L = 0, 0, len(w)
        while i < L and self._cons(w, i):
            i += 1
        while i < L:
            while i < L and not self._cons(w, i):
                i += 1
            if i >= L:
                break
            n += 1
            while i < L and self._cons(w, i):
                i += 1
        return n

    def _vowel(self, w):
        return any(not self._cons(w, i) for i in range(len(w)))

    def _dbl(self, w):
        return len(w) >= 2 and w[-1] == w[-2] and self._cons(w, len(w) - 1)

    def _cvc(self, w):
        return (len(w) >= 3 and self._cons(w, len(w) - 1) and not self._cons(w, len(w) - 2)
                and self._cons(w, len(w) - 3) and w[-1] not in "wxy")

    def _rep(self, w, suf, rep, cond=lambda s: True):
        if w.endswith(suf):
            s = w[: len(w) - len(suf)]
            if cond(s):
                return s + rep, True
            return w, True
        return w, False

    def stem(self, word):
        w = word.lower()
        if len(w) <= 2:
            return w
        # step 1a
        for suf, rep in (("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")):
            if w.endswith(suf):
                w = w[: len(w) - len(suf)] + rep
                break
        # step 1b
        if w.endswith("eed"):
            if self._m(w[:-3]) > 0:
                w = w[:-1]
        else:
            done = False
            for suf in ("ed", "ing"):
                if w.endswith(suf) and self._vowel(w[: len(w) - len(suf)]):
                    w = w[: len(w) - len(suf)]
                    done = True
                    break
            if done:
                if w.endswith(("at", "bl", "iz")):
                    w += "e"
                elif self._dbl(w) and w[-1] not in "lsz":
                    w = w[:-1]
                elif self._m(w) == 1 and self._cvc(w):
                    w += "e"
        # step 1c
        if w.endswith("y") and self._vowel(w[:-1]):
            w = w[:-1] + "i"
        # step 2
        for suf, rep in (("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
                         ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
                         ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
                         ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
                         ("logi", "log")):
            if w.endswith(suf):
                if self._m(w[: len(w) - len(suf)]) > 0:
                    w = w[: len(w) - len(suf)] + rep
                break
        # step 3
        for suf, rep in (("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"),
                         ("ful", ""), ("ness", "")):
            if w.endswith(suf):
                if self._m(w[: len(w) - len(suf)]) > 0:
                    w = w[: len(w) - len(suf)] + rep
                break
        # step 4
        for suf in ("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou",
                    "ism", "ate", "iti", "ous", "ive", "ize"):
            if w.endswith(suf):
                s = w[: len(w) - len(suf)]
                if self._m(s) > 1 and (suf != "ion" or s[-1:] in ("s", "t")):
                    w = s
                break
        # step 5a
        if w.endswith("e"):
            s = w[:-1]
            if self._m(s) > 1 or (self._m(s) == 1 and not self._cvc(s)):
                w = s
        # step 5b
        if self._m(w) > 1 and self._dbl(w) and w.endswith("l"):
            w = w[:-1]
        return w

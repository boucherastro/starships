class list_of_dict(list):

    def get_all(self, *key, ind=slice(None)):
        '''
        Returns a list of the key along the list of dict
        '''

        out = ()
        for k in key:
            out += ([dic[k] for dic in self[ind]],)

        return out

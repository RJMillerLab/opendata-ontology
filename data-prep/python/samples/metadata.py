import os
import json

def get_metadata(tablename, labels):
    print(os.path.join(os.path.dirname(tablename), "ckan.json"))
    if os.path.isfile(os.path.join(os.path.dirname(tablename), "ckan.json")):
        return ckan_metadata(os.path.join(os.path.dirname(tablename), "ckan.json"), labels)
    if os.path.isfile(os.path.join(os.path.dirname(tablename), "socrata.json")):
        return socrata_metadata(os.path.join(os.path.dirname(tablename), "socrata.json"), labels)
    return (None, labels)

def ckan_metadata(metadatafile, labels):
    str_labels = []
    data = json.load(open(metadatafile))
    #if not isinstance(data["title"], dict):
    #    title = data["title"].lower()
    #else:
    #    title = data["title"]["en"].lower()
    #str_labels.append("ckan_title_" + title)
    #if data["notes"] is not None:
    #    str_labels.append("ckan_notes_" + data["notes"].lower())
    #str_labels.append("ckan_name_" + data["name"].lower())
    #str_labels.append("ckan_id_" + data["id"].lower())
    #str_labels.append("ckan_organizationtitle_" + data["organization"]["title"].lower())
    #str_labels.append("ckan_organizationname_" + data["organization"]["name"].lower())
    if "topic_category" in data:
        for c in data["topic_category"]:
            str_labels.append("ckan_topiccategory_" + c.lower())
    if "keywords" in data:
        for k in data["keywords"]["en"]:
            str_labels.append("ckan_keywords_" + k.lower())
    if "tags" in data:
        for t in data["tags"]:
            str_labels.append("ckan_tags_" + t["name"].lower())
    if "subject" in data:
        for s in data["subject"]:
            str_labels.append("ckan_subject_" + s.lower())
    #for res in data["resources"]:
    #    str_labels.append("ckan_position_" + str(res["position"]))
    #    str_labels.append("ckan_id_" + str(res["id"]))
    #    str_labels.append("ckan_url_" + str(res["url"]))
    #    str_labels.append("ckan_name_" + str(res["name"]))
    #    if "package_id" in res:
    #        str_labels.append("ckan_packageid_" + str(res["package_id"]))
    t_labels = []
    for l in str_labels:
        if l in labels:
            if labels[l] not in t_labels:
                t_labels.append(labels[l])
        else:
            t_labels.append(len(labels))
            labels[l] = len(labels)
    return (t_labels, labels)

def socrata_metadata(metadatafile, labels):
    str_labels = []
    data = json.load(open(metadatafile))
    #str_labels.append("socrata_link_" + data["link"].lower())
    #str_labels.append("socrata_id_" + data["resource"]["id"].lower())
    #str_labels.append("socrata_name_" + data["resource"]["name"].lower())
    #str_labels.append("socrata_description_" + data["resource"]["description"].lower())
    #str_labels.append("socrata_domain_" + data["metadata"]["domain"].lower())
    if 'classification' in data:
        if "domain_category" in data["classification"]:
            if data["classification"]["domain_category"] is not None:
                str_labels.append("socrata_domaincategory_" + data["classification"]["domain_category"].lower())
        for t in data["classification"]["domain_tags"]:
            str_labels.append("socrata_domaintags_" + t.lower())
        for t in data["classification"]["tags"]:
            str_labels.append("socrata_tags_" + t.lower())
    # find the index of labels
    t_labels = []
    for l in str_labels:
        if l in labels:
            if labels[l] not in t_labels:
                t_labels.append(labels[l])
        else:
            t_labels.append(len(labels))
            labels[l] = len(labels)
    return (t_labels, labels)



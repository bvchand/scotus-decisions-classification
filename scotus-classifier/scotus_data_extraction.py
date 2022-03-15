import textacy.datasets
import pandas as pd


def extract_data():
    ds = textacy.datasets.SupremeCourt()
    ds.download()

    decisions = ds.records()
    scotus_dataset = pd.DataFrame(columns=['scotus_issue_area', 'scotus_issue', 'scotus_text'])
    # issue_freq = {}
    # issue_area_freq = {}

    for text, details in decisions:
        issue_area = details['issue_area']
        issue = details['issue']
        if details['issue_area'] == -1:
            continue
        if details['issue'] == 'none':
            continue

        scotus_dataset = scotus_dataset.append(
            {'scotus_issue_area': issue_area, 'scotus_issue': issue, 'scotus_text': text}, ignore_index=True)

    scotus_dataset.to_csv('/Users/bharathi/PythonWorkspace/scotus_decisions_application/data/full_scotus.csv', index=True)
    scotus_sample = scotus_dataset.sample(n=1500, replace=False, weights=scotus_dataset['scotus_issue_area'],
                                        random_state=4)

    scotus_sample.to_csv('/Users/bharathi/PythonWorkspace/scotus_decisions_application/data/scotus_sample.csv', index=True)
    #     if details['issue'] in issue_freq:
    #         issue_freq[details['issue']] = issue_freq[details['issue']] + 1
    #     else:
    #         issue_freq[details['issue']] = 1
    #
    #     if details['issue_area'] in issue_area_freq:
    #         issue_area_freq[details['issue_area']] = issue_area_freq[details['issue_area']] + 1
    #     else:
    #         issue_area_freq[details['issue_area']] = 1
    #
    # plot_issue_freq(issue_freq, issue_area_freq)
    return scotus_dataset, scotus_sample

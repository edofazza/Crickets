import openai

if __name__ == '__main__':
    openai.api_key = "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"
    model_engine = 'text-davinci-003'
    prompt = 'What should be included in the discussion and conclusions parts for a research article?'
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = completion.choices[0].text
    print(response)

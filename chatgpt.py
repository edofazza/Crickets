import openai

if __name__ == '__main__':
    openai.api_key = "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"
    model_engine = 'text-davinci-003'
    prompt = 'Can you write better and more formally the following paragraph needed for a scientific article: This study demonstrates the importance of artificial intelligence in the advancement of animal biosensor in order to have an automated mechanism that requires only video recordings and can read information directly using the tracking sequences of the animals. Herein, we propose a novel workflow to develop IABS based on state-of-the art pose estimation and sequence processing techniques that can be easily extended to other similar tasks, e.g., using different animals and different stimuli. Overall, our results show how this workflow can lay the foundations for new research directions towards the development of computer vision based biosensor systems that are not invasive to the animal and can be easily trained and deployed.'
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

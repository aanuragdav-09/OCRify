import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

test("updates text statistics as the extracted text is edited", () => {
	render(<App />);

	fireEvent.change(screen.getByRole("textbox", { name: "Extracted text" }), {
		target: { value: "Hello World!\n2024" },
	});

	expect(screen.getByText("Total Characters").nextSibling).toHaveTextContent("17");
	expect(screen.getByText("Characters (excluding spaces)").nextSibling)
		.toHaveTextContent("15");
	expect(screen.getByText("Total Words").nextSibling).toHaveTextContent("3");
	expect(screen.getByText("Total Lines").nextSibling).toHaveTextContent("2");
	expect(screen.getByText("Total Numbers").nextSibling).toHaveTextContent("1");
});

test("reports spelling differences against reference text", () => {
	render(<App />);

	fireEvent.change(screen.getByRole("textbox", { name: "Extracted text" }), {
		target: { value: "Hello wurld" },
	});
	fireEvent.change(screen.getByRole("textbox", { name: "Reference Text" }), {
		target: { value: "Hello world" },
	});

	expect(screen.getByText("Character Match").nextSibling).toHaveTextContent("91%");
	expect(screen.getByText("Word Match").nextSibling).toHaveTextContent("50%");
	expect(screen.getByText("world → wurld")).toBeInTheDocument();
});

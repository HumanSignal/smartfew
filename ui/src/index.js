import React from 'react';
import ReactDOM from 'react-dom';
import {observer} from "mobx-react";
import {types, flow, getSnapshot} from "mobx-state-tree";
import 'antd/dist/antd.css';
import {Button, Row, Col} from "antd";


const ImageItem = types
    .model({
        url: types.string,
        selected: types.optional(types.boolean, false)
    })
    .actions(self => ({
        toggleSelected() {
            self.selected = !self.selected
        },
    }));

const ImageItemView = observer(({ item }) => (
    <p>
        {item.url && <img
            src={item.url}
            onClick={() => item.toggleSelected()}
            style={{
                padding: "1em",
                maxWidth: "100%",
                border: item.selected ? "5px solid red" : "1px",
            }}
        />}
    </p>
));

const ImageItems = types
    .model({
        images: types.optional(types.array(ImageItem), [])
    })
    .actions(self => {
        const messageMe = function () {
            alert(JSON.stringify(getSnapshot(self)));
        };
        const loadImages = flow(function* loadImages() {
            const response = yield window.fetch('http://localhost:14321/update', {
                method: 'post',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(getSnapshot(self))
            });

            const r = yield response.json();
            self.images = r.images.map(i => ImageItem.create(i));
        });

        return {
            loadImages,
            messageMe
        }
    });

const ImageItemsView = observer(({ items }) => {
    let rows = [];
    items.images.forEach((image, index) => {
        rows.push(
            <Col span={6}>
                <ImageItemView item={image}/>
            </Col>
        )
    });
    return (
        <div>
        <Row>
            {rows}
        </Row>
        </div>
    )
});


const imageItems = ImageItems.create({
    images: []
});


class App extends React.Component {
    render() {
        let numImages = this.props.imageItems.length;
        let buttonName = numImages > 0 ? "Submit" : "Start";
        return (
            <div>
                <ImageItemsView items={this.props.imageItems}/>
                <div align="center">
                    <Button type="primary" size="large" onClick={this.props.imageItems.loadImages}>{buttonName}</Button>
                </div>
            </div>
        )
    }
}


ReactDOM.render(
  <App imageItems={imageItems}/>,
  document.getElementById('root')
);
